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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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
    from ..services.hub_cache import load_player_name_map

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
    """The model's own win/loss call, settled against real completed games
    from pre-tip picks only (services/hub_service._settle_game_outcome)."""
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


def _line_from_market(row: Any, team: str) -> str | None:
    """A pre-tip market row's own line, written with its team ('BOS -3.5')."""
    point = _num(row["point"])
    if point is None:
        return None
    if point == 0:
        return f"{team} PK"
    number = f"{abs(point):.0f}" if float(point).is_integer() else f"{abs(point):.1f}"
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
            market["line"] = quoted
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




def _context(game: dict, schedule: list[dict]) -> dict:
    """Rest and back-to-back, but only when the schedule already carries
    them. No new features are computed here."""
    context: dict[str, Any] = {}
    home_rest = game.get("home_rest_days")
    away_rest = game.get("away_rest_days")
    if home_rest is not None and away_rest is not None:
        context["rest"] = f"Rest {home_rest} v {away_rest} days"
    if game.get("home_back_to_back") is not None:
        context["back_to_back"] = bool(game.get("home_back_to_back") or game.get("away_back_to_back"))
    return context


def _record() -> dict | None:
    row = _track_record()
    if not row:
        return None
    settled = int(row.get("total_predictions") or 0)
    if settled <= 0:
        return None
    return {
        "label": "Picks made before tip-off",
        "hits": int(row.get("correct_predictions") or 0),
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
        "context": _context(game, schedule),
        "players": _players(str(game_id), {home_team, away_team}),
        "record": _record(),
        "result": _result(game, status, pick_timing, home_prob),
    }
