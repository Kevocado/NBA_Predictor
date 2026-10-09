"""facts.py — the read-only NBA /facts bundle the match explainer consumes.

Read-only and suggest-only: this router never writes, never trains and
never places anything.

NBA has no public snapshot of predictions: every number comes from the same
SQLite store the site itself reads, and ``tracking/timing.py`` is the single
authority on whether a pick was made before tip-off. That makes the honesty
rules cheap to honour, and they are the same rules the other sports follow:

* a pick for a game that has started comes only from the newest row made
  before tip-off, never from a later backfill row;
* no pre-tip row means the pick is ``rebuilt`` — shown, never judged;
* no prediction at all means ``pick`` is null and ``pick_timing`` is ``none``;
* ``result.pick_won`` appears only for a ``pre_kickoff`` pick.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from .. import config
from ..api import deps
from ..services import hub_service
from ..services.schedule_repository import get_game
from ..tracking import store
from ..tracking.timing import latest_pre_tip, made_before_tip, pick_cutoff

router = APIRouter()
logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _schedule() -> list[dict]:
    return deps.get_schedule(deps.get_schedule_path())


#: How far back the duels' history reaches. A team plays ~3 games a week and a
#: duel reads its last 15, so five weeks covers a full window with room for a
#: team that has played less. Ranking further back than that reads a form the
#: team has since moved on from.
BOX_LOOKBACK_DAYS = 120

_BOX_HISTORY: dict = {}

#: How many `as_of` windows to keep. A game page asks for one and the explainer's
#: pre-generation walks the week's games, so a handful covers a full pass; more
#: than that and the oldest entries have gone cold anyway.
BOX_HISTORY_KEEP = 8

#: How often the warmer checks the schedule cache. Same reasoning as the odds
#: refresher's tick: a faster one spends no requests when nothing has changed,
#: because both readers are cached files.
BOX_WARM_POLL_SECONDS = float(os.getenv("BOX_WARM_POLL_SECONDS", "30"))


def _build_box_history(as_of: str) -> pd.DataFrame:
    """Build the lookback window of completed games with their box scores.

    `enrich_with_boxscores` is the pipeline's own step that adds the flat
    ``home_*/away_*`` box-score fields, and `to_training_frame` is its own
    shaping step -- so the duels rank exactly the figures the features are built
    from, rather than a second, separately-fetched copy that could disagree.

    **Bounded.** The ranker needs each team's last `window` games, so only
    completed games in a lookback window ending at `as_of` are enriched. Reading
    every completed game in the schedule would put a season of ESPN calls on a
    single request.

    Only ever called from `warm_box_history`, which runs on a daemon thread.
    """
    from ..pipeline.ingest import enrich_with_boxscores, to_training_frame

    start = (pd.Timestamp(as_of) - pd.Timedelta(days=BOX_LOOKBACK_DAYS)).date().isoformat()
    window = [
        g for g in _schedule()
        if start <= str(g.get("game_date", "")) < str(as_of) and g.get("completed")
    ]
    return to_training_frame(enrich_with_boxscores(window))


def _box_score_history(as_of: str) -> pd.DataFrame:
    """The warmed history for `as_of`, or an empty frame when it is not warm.

    **Never builds.** Building reads box scores, which means I/O, and I/O on a
    request thread is what puts a season of ESPN calls on the first visitor.
    `warm_box_history` is what builds, from a daemon thread started at the app's
    lifespan, for every upcoming date -- so a request that arrives before the
    warm finishes shows no matchup duels rather than making its visitor wait.

    An empty frame is read as "no data, no signal", the same rule the signals
    endpoint follows. It is logged, because a warmer that silently never
    succeeds is a cold path nobody would otherwise be told about.

    Cached on ``(schedule state, as_of)``. The schedule cache is what gains rows
    as games complete, so its (mtime, size) is the honest invalidation signal --
    the same shape `routes._db_state` uses for the MAE cache, and an unchanged
    file costs one stat().
    """
    state = _schedule_state(deps.get_schedule_path())
    key = (state, str(as_of))
    frame = _BOX_HISTORY.get(key)
    if frame is None:
        logger.info(
            "box-score history for %s is not warm; no matchup duels on this request", as_of
        )
        return pd.DataFrame()
    return frame


def _schedule_state(path: Path) -> tuple[int, int]:
    """(mtime_ns, size) of the schedule cache, or (-1, -1) when unreadable."""
    try:
        st = Path(path).stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return (-1, -1)


def warm_box_history(as_of: str) -> bool:
    """Build `as_of`'s window and store it, off the request path. Never raises.

    Same contract as `routes.warm_mae_cache`: a warm that fails must not take the
    API down, and must not overwrite a previous window with a broken one. It logs
    instead, because a warm that silently never succeeds is a cold path nobody is
    told about.
    """
    try:
        frame = _build_box_history(as_of)
    except Exception as exc:  # noqa: BLE001 - reported, not raised: see docstring
        logger.error("box-score history warm failed for %s: %s: %s", as_of, type(exc).__name__, exc)
        return False
    _store_box_history(as_of, frame)
    return True


def _store_box_history(as_of: str, frame: pd.DataFrame) -> None:
    """Put `frame` in the cache for `as_of`, evicting the coldest entry."""
    state = _schedule_state(deps.get_schedule_path())
    _BOX_HISTORY[(state, str(as_of))] = frame
    # dicts keep insertion order, so the first key is the coldest one.
    while len(_BOX_HISTORY) > BOX_HISTORY_KEEP:
        del _BOX_HISTORY[next(iter(_BOX_HISTORY))]


def start_box_history_warmer(
    schedule_path: Path,
    *,
    clock=time.monotonic,
    sleep=time.sleep,
    should_stop=None,
) -> threading.Thread:
    """Warm the next `BOX_HISTORY_KEEP` upcoming dates -- as many as the cache
    holds -- then re-warm when the schedule cache changes or a build failed.

    Only as many as the cache holds: a date warmed and then evicted before it is
    served is no better than one never warmed, so the bound and the warm list have
    to agree.

    A date whose build failed is retried on the next tick rather than left cold
    until the schedule happens to change under it.

    A daemon thread started from the app's lifespan: serving begins whether or
    not the warm finishes, and a warm that hangs cannot hold shutdown either.
    ``clock``, ``sleep`` and ``should_stop`` are injected so the loop is testable
    without waiting on a real tick.

    The invalidation signal is the schedule cache's (mtime, size) -- the file that
    gains a row each time a game completes, which is the moment the duels' answer
    changes. An unchanged file costs one stat() per tick and nothing else.
    """
    schedule_path = Path(schedule_path)

    def _run() -> None:
        seen_state: tuple[int, int] | None = None
        pending: set[str] = set()
        while should_stop is None or not should_stop():
            state = _schedule_state(schedule_path)
            if state != seen_state:
                # A schedule state this pass has not covered yet: warm the next
                # `BOX_HISTORY_KEEP` dates. Only as many as the cache holds -- a
                # date warmed and then evicted before it is served is no better
                # than one never warmed, so the bound and the warm list have to
                # agree -- plus anything still pending from before.
                schedule = _schedule()
                upcoming = sorted({
                    str(g["game_date"]) for g in schedule
                    if not g.get("completed") and g.get("game_date")
                })
                if not upcoming:
                    logger.info("no upcoming games to warm duel history for")
                dates = set(upcoming[:BOX_HISTORY_KEEP]) | pending
                seen_state = state
            elif pending:
                # Same schedule, but a date failed last time: retry only that.
                # Re-warming the dates that already succeeded would rebuild them
                # for nothing.
                dates = set(pending)
            else:
                dates = set()
            pending = set()
            for date in sorted(dates):
                if not warm_box_history(date):
                    pending.add(date)
            sleep(BOX_WARM_POLL_SECONDS)

    thread = threading.Thread(target=_run, name="box-history-warmer", daemon=True)
    thread.start()
    return thread


def _matchup_rows(home: str, away: str, as_of, pick_side: str | None) -> list[dict]:
    """Four-factors rank duels for this game, in the facts bundle's shape.

    Delegates to `signals.four_factors_duel`, which owns the window, the
    `min_gap` and the ranking. Empty when either team has not played enough
    games to be ranked -- a duel off last month's roster is a number, not a
    measurement.
    """
    from ..signals.four_factors_duel import four_factors_duel, to_context

    history = _box_score_history(as_of)
    if not len(history):
        return []
    duels = four_factors_duel(home, away, history, as_of)
    return to_context(duels, pick_side)


def _db_path() -> Path:
    return deps.get_db_path()


# --- store access (indirected so tests never open a real database) -------

def _predictions(game_id: str) -> list:
    return list(store.get_predictions_for_game(_db_path(), game_id))


def _market_rows(game_id: str) -> list:
    return list(store.get_market_predictions_for_game(_db_path(), game_id))


def _player_rows(game_id: str) -> list:
    """The same per-player/stat pick the site's /games/{id}/players uses:
    the newest row before tip-off, else the latest (a rebuilt backtest),
    flagged so a rebuilt projection is never presented as a real one."""
    from ..services.hub_service import load_player_name_map

    game = get_game(_schedule(), game_id)
    if game is None:
        return []
    name_by_id = load_player_name_map(config.DATA_DIR / "cache" / "hub" / "players.json")
    by_key: dict[tuple[str, str], list] = {}
    for row in store.get_player_predictions_for_game(_db_path(), game_id):
        by_key.setdefault((row["player_id"], row["stat"]), []).append(row)
    out = []
    for (player_id, stat_name), rows in by_key.items():
        pick = latest_pre_tip(rows, game)
        rebuilt = pick is None
        pick = pick if pick is not None else max(rows, key=lambda r: r["created_at"])
        out.append({
            "player_id": player_id,
            "player_name": name_by_id.get(player_id, player_id),
            "stat": stat_name,
            "predicted_value": pick["predicted_value"],
            "rebuilt": rebuilt,
        })
    return out


def _track_record() -> dict | None:
    """The model's own win/loss call against real completed games.

    Served whole, headline and pre-tip subset both, because two call sites read
    it for two different questions: `_record()` takes the `pre_tip` sub-record
    (its block is labelled "Picks made before tip-off") and nothing else may.
    See services/hub_service.compute_track_record for which picks count.
    """
    for row in hub_service.compute_track_record(_db_path(), _schedule()):
        if getattr(row, "market", None) == "game_outcome":
            return row.model_dump() if hasattr(row, "model_dump") else dict(row)
    return None


# --- helpers ------------------------------------------------------------

def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _iso_utc(value: Any) -> str:
    if not value:
        return ""
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _as_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _tip_off(game: dict) -> datetime | None:
    """The site's own cutoff: tip_off when known, else noon Eastern on the
    game date (timing.pick_cutoff)."""
    try:
        return pick_cutoff(game)
    except (KeyError, TypeError, ValueError):
        return _as_utc(game.get("tip_off"))


def _status(game: dict, now: datetime) -> str:
    if game.get("completed") or (game.get("home_pts") is not None and game.get("away_pts") is not None):
        return "final"
    tip = _tip_off(game)
    return "live" if tip is not None and tip <= now else "upcoming"


def _favourite(home_team: str, away_team: str, home_prob: Any) -> dict | None:
    home = _num(home_prob)
    if home is None:
        return None
    # Home wins a 50/50 tie, matching the site's favourite()/pickWon rule.
    if home >= 0.5:
        return {"label": home_team, "prob": home}
    return {"label": away_team, "prob": 1.0 - home}


def _margin_line(home_team: str, away_team: str, home_prob: float, margin: float) -> str:
    """Port of frontend/src/lib/pick.ts marginLine: the projected margin is
    written so it can never contradict the pick. The win and margin numbers
    come from separate models, so when they point at different teams — or
    the margin rounds to nothing — this says 'Toss-up' rather than printing
    'BOS to win' beside 'MIA by 0.6'."""
    pick_team = home_team if home_prob >= 0.5 else away_team
    team = home_team if margin >= 0 else away_team
    value = abs(margin)
    if team == pick_team and value >= 0.5:
        return f"{team} by {value:.1f}"
    return "Toss-up"


def _line_from_market(row: Any, team: str, market: str | None = None) -> str | None:
    """A pre-tip market row's own line, written the way a bettor reads it.

    A spread is signed and team-attributed ('BOS -3.5'); a total is neither
    ('Over 224.5'). Reusing the spread wording for a totals row would render
    'Over +224.5', which is not a line anyone would say.
    """
    point = _num(row["point"])
    if point is None:
        return None
    number = f"{abs(point):.0f}" if float(point).is_integer() else f"{abs(point):.1f}"
    if (market or row["market"]) == "totals":
        return f"{team} {number}"
    if point == 0:
        return f"{team} PK"
    return f"{team} {'-' if point < 0 else '+'}{number}"


def _market_line(game: dict, market: str, default_team: str | None) -> str | None:
    """The newest pre-tip row's line for this market. Rows made after tip-off
    are ignored: quoting a post-tip line for a started game would be exactly
    the kind of hindsight this bundle exists to avoid."""
    best = None
    for row in _market_rows(game["game_id"]):
        if row["market"] != market or not made_before_tip(row["created_at"], game):
            continue
        if best is None or row["created_at"] > best["created_at"]:
            best = row
    if best is None:
        return None
    team = default_team or best["selection"]
    return _line_from_market(best, team)


def _markets(game: dict, prediction: dict | None) -> list[dict]:
    home_team, away_team = game["home_team"], game["away_team"]
    out: list[dict] = []
    if prediction is None:
        return out

    home_prob = _num(prediction.get("home_win_prob"))
    if home_prob is not None:
        out.append({
            "market": "moneyline",
            "model": {home_team: home_prob, away_team: 1.0 - home_prob},
        })

    margin = _num(prediction.get("predicted_margin"))
    if margin is not None and home_prob is not None:
        pick = _favourite(home_team, away_team, home_prob)
        market: dict[str, Any] = {
            "market": "spread",
            "model_margin": margin,
            "line": _margin_line(home_team, away_team, home_prob, margin),
        }
        quoted = _market_line(game, "spread", pick["label"] if pick else home_team)
        if quoted is not None:
            market["market_line"] = quoted
        out.append(market)

    total = _num(prediction.get("predicted_total"))
    if total is not None:
        market = {"market": "total", "model_total": total}
        quoted = _market_line(game, "totals", None)
        if quoted is not None:
            # `market_line` in every market, so the panel reads one key for the
            # quoted book line. `line` is reserved for the model's own wording
            # (the spread's marginLine), which is a different thing.
            market["market_line"] = quoted
        out.append(market)
    return out


def _players(game_id: str, teams: set[str]) -> list[dict]:
    # The id is passed, never kept in module state: sync endpoints run in a
    # thread pool, and a shared "current game" let one request quote
    # another game's players.
    rows = [r for r in _player_rows(game_id) if not r["rebuilt"]]
    rows.sort(key=lambda r: _num(r.get("predicted_value")) or 0.0, reverse=True)
    out = []
    for row in rows[:3]:
        value = _num(row.get("predicted_value"))
        stat_name = str(row.get("stat") or "").replace("_", " ")
        out.append({
            "name": row.get("player_name"),
            "team": row.get("team") if row.get("team") in teams else None,
            "projection": f"{value:.1f} {stat_name}" if value is not None else stat_name,
        })
    return out




def _context(game: dict, schedule: list[dict], started: bool = False, pick: dict | None = None) -> dict:
    """Rest, back-to-back and the four-factors rank duels.

    Rest and back-to-back come from the schedule and only when it already
    carries them. The duels are computed here from box scores, because there is
    nowhere else for them to come from: the schedule repository keeps only
    scores, so `_box_score_history` re-reads the box fields.
    """
    context: dict[str, Any] = {}
    home_rest = game.get("home_rest_days")
    away_rest = game.get("away_rest_days")
    if home_rest is not None and away_rest is not None:
        context["rest"] = f"Rest {home_rest} v {away_rest} days"
    if game.get("home_back_to_back") is not None:
        context["back_to_back"] = bool(game.get("home_back_to_back") or game.get("away_back_to_back"))

    if not started:
        home, away = game["home_team"], game["away_team"]
        pick_side = None
        if pick:
            # The duel's `toward` is "home"/"away", so the pick has to be read as
            # a side rather than as the team name `_favourite` returns.
            pick_side = "home" if pick.get("label") == home else "away" if pick.get("label") == away else None
        try:
            matchups = _matchup_rows(home, away, game["game_date"], pick_side)
        except Exception:
            # A duel is an enhancement on a game page, and the page has to
            # survive its absence -- the same rule the signals endpoint follows.
            logger.exception("matchup duels unavailable for %s", game.get("game_id"))
            matchups = []
        if matchups:
            context["matchups"] = matchups
    return context


def _record() -> dict | None:
    """The PRE-TIP record, under a label that says so.

    Since 2026-10-01 the track record's headline counts every recorded pick,
    including ones the model made after tip-off (`track-record-counts-every-
    pick`, merged as predictor-hub #66). So this must read the `pre_tip`
    sub-record, not the headline: the number on this block is labelled "Picks
    made before tip-off", and putting an all-picks figure under that label is
    exactly the mislabelling the spec still forbids ("nothing computed after
    the start may be labelled 'made before kickoff'").

    The pre-tip subset is what this block showed before the rule changed, so
    the figures in the explainer are unchanged. A payload with no `pre_tip` at
    all yields None rather than a mislabelled record -- an omission is
    recoverable, a wrong label is not.
    """
    row = _track_record()
    if not row:
        return None
    before_tip = row.get("pre_tip")
    if not isinstance(before_tip, dict):
        return None
    settled = int(before_tip.get("total_predictions") or 0)
    if settled <= 0:
        return None
    return {
        "label": "Picks made before tip-off",
        "hits": int(before_tip.get("correct_predictions") or 0),
        "settled": settled,
    }


def _result(game: dict, status: str, pick_timing: str, home_prob: Any) -> dict | None:
    if status != "final":
        return None
    home_pts = _num(game.get("home_pts"))
    away_pts = _num(game.get("away_pts"))
    if home_pts is None or away_pts is None:
        return None
    result: dict[str, Any] = {"score": f"{game['home_team']} {home_pts:.0f}-{away_pts:.0f}"}
    if pick_timing != "pre_kickoff":
        return result
    probability = _num(home_prob)
    if probability is None:
        return result
    # The site's own rule: home at >= 50% is the pick.
    result["pick_won"] = bool((probability >= 0.5) == (home_pts > away_pts))
    return result


@router.get("/facts/upcoming")
def get_facts_upcoming(hours: int = 72) -> dict:
    """Ids of games tipping off within the window, for pre-generation."""
    if hours < 0:
        raise HTTPException(status_code=422, detail="hours must be >= 0")
    now = _now()
    cutoff = now + timedelta(hours=hours)
    ids = []
    for game in _schedule():
        tip = _tip_off(game)
        if tip is None or tip <= now or tip > cutoff:
            continue
        game_id = game.get("game_id")
        if game_id and str(game_id) not in ids:
            ids.append(str(game_id))
    return {"ids": ids}


@router.get("/facts/{game_id}")
def get_facts(game_id: str) -> dict:
    now = _now()
    schedule = _schedule()
    game = get_game(schedule, game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game_id: {game_id}")

    home_team, away_team = game["home_team"], game["away_team"]
    status = _status(game, now)
    started = status in ("live", "final")

    rows = _predictions(game_id)
    pre_tip = latest_pre_tip(rows, game) if rows else None
    chosen = pre_tip
    if chosen is None and rows and not started:
        # An upcoming game can still show today's newest read.
        chosen = max(rows, key=lambda r: r["created_at"])

    pick = _favourite(home_team, away_team, chosen["home_win_prob"]) if chosen is not None else None
    if pick is None:
        pick_timing = "none"
    elif pre_tip is not None:
        pick_timing = "pre_kickoff"
    else:
        pick_timing = "rebuilt"

    prediction = dict(chosen) if chosen is not None else None
    home_prob = prediction.get("home_win_prob") if prediction else None

    return {
        "sport": "nba",
        "id": str(game_id),
        "title": f"{away_team} at {home_team}",
        "starts_at": _iso_utc(_tip_off(game)),
        "status": status,
        "pick_timing": pick_timing,
        "pick": pick,
        "markets": [] if (started and chosen is None) else _markets(game, prediction),
        "drivers": [],
        "context": _context(game, schedule, started=started, pick=pick),
        "players": _players(str(game_id), {home_team, away_team}),
        "record": _record(),
        "result": _result(game, status, pick_timing, home_prob),
    }
