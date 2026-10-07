import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from nba_predictor.api.availability import resolve_doubtful_players, resolve_out_players
from nba_predictor.models.probability import cover_prob
from nba_predictor.models.manifest import build_manifest
from nba_predictor.api.deps import (
    get_db_path,
    get_injury_report,
    get_injury_report_best_effort,
    get_models_dir,
    get_schedule,
    get_schedule_path,
    get_today,
    get_training_games_path,
    require_admin,
)
from nba_predictor.api.schemas import (
    DoubtfulPlayerOut,
    GameDetailOut,
    GameOut,
    HeadToHeadMeetingOut,
    MarketPredictionOut,
    OutPlayerOut,
    PlayerPropOut,
    PredictionOut,
    SeasonBoundsOut,
    TeamOut,
    TrackRecordOut,
    VsMarketOut,
)
from nba_predictor.models.player_props import in_sample_mae_by_stat
from nba_predictor.data.team_reference import TEAMS, get_team
from nba_predictor.odds.value_bets import std_from_mae
from nba_predictor.pipeline.ingest import run_ingest
from nba_predictor.odds import edge_gate
from nba_predictor.odds.edge_gate import _is_fresh
from nba_predictor.pipeline.refresh_odds import refresh_market_predictions
from nba_predictor.pipeline.retrain import run_retrain_pipeline
from nba_predictor.services.calibration_service import compute_model_calibration
from nba_predictor.services.hub_service import (
    compute_track_record,
    compute_vs_market,
    load_hub_cache,
    load_player_name_map,
    warm_player_props_cache,
)
from nba_predictor.services.schedule_repository import (
    default_week_start,
    get_game,
    get_games_for_date,
    get_games_for_week,
    get_head_to_head,
    get_recent_form,
    load_schedule,
)
from nba_predictor.tracking import store
from nba_predictor import config
from nba_predictor.tracking.store import get_recent_market_predictions
from nba_predictor.tracking.player_props import (
    actuals_by_player_stat,
    picks_by_player_stat,
    resolved_player_props,
)
from nba_predictor.tracking.timing import latest_by_instant, latest_pre_tip, made_before_tip
from nba_predictor import config

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/teams", response_model=list[TeamOut])
def list_teams() -> list[TeamOut]:
    return [
        TeamOut(abbreviation=t.abbreviation, name=t.name, conference=t.conference, division=t.division)
        for t in TEAMS
    ]


@router.get("/teams/{abbreviation}", response_model=TeamOut)
def get_team_detail(abbreviation: str) -> TeamOut:
    try:
        team = get_team(abbreviation)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown team: {abbreviation}")
    return TeamOut(abbreviation=team.abbreviation, name=team.name, conference=team.conference, division=team.division)


def _cover_and_sigma_from_manifest(manifest: dict | None) -> tuple[float | None, float | None, float | None]:
    """(margin_sigma, total_sigma, win_auc) from the manifest, or None if not present."""
    if not manifest:
        return None, None, None
    margin = manifest.get("metrics", {}).get("margin", {})
    total = manifest.get("metrics", {}).get("total", {})
    win = manifest.get("metrics", {}).get("win_probability", {})
    return margin.get("residual_sigma"), total.get("residual_sigma"), win.get("auc")


def _prediction_out(db_path: Path, g: dict, manifest: dict | None = None) -> tuple[PredictionOut | None, bool]:
    """The pick to show and whether it was rebuilt after tip-off. The latest
    pick made before tip-off wins; a later backtest row only shows (labelled
    rebuilt) when no pre-tip pick exists."""
    rows = store.get_predictions_for_game(db_path, g["game_id"])
    if not rows:
        return None, False
    row = latest_pre_tip(rows, g)
    rebuilt = row is None
    row = row if row is not None else latest_by_instant(rows)
    margin_sigma, total_sigma, _ = _cover_and_sigma_from_manifest(manifest)
    return (
        PredictionOut(
            home_win_probability=row["home_win_prob"],
            predicted_margin=row["predicted_margin"],
            predicted_total=row["predicted_total"],
            cover_prob_spread=cover_prob(row["predicted_margin"], row["market_point"] if "market_point" in row.keys() else None, margin_sigma) if margin_sigma is not None else None,
            cover_prob_total=cover_prob(row["predicted_total"], row["market_total_point"] if "market_total_point" in row.keys() else None, total_sigma) if total_sigma is not None else None,
            margin_sigma=margin_sigma,
            total_sigma=total_sigma,
        ),
        rebuilt,
    )


def load_latest_manifest() -> dict | None:
    path = config.PROJECT_ROOT / "models" / "manifest.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _game_out(g: dict, db_path: Path) -> GameOut:
    manifest = load_latest_manifest()
    prediction, rebuilt = _prediction_out(db_path, g, manifest)
    return GameOut(
        game_id=g["game_id"], game_date=g["game_date"], tip_off=g.get("tip_off"),
        home_team=g["home_team"], away_team=g["away_team"],
        prediction=prediction, rebuilt=rebuilt,
        completed=g.get("completed", False), home_pts=g.get("home_pts"), away_pts=g.get("away_pts"),
    )


@router.get("/games", response_model=list[GameOut])
def list_games(date: str, schedule: list[dict] = Depends(get_schedule), db_path: Path = Depends(get_db_path)) -> list[GameOut]:
    games = get_games_for_date(schedule, date)
    return [_game_out(g, db_path) for g in games]


@router.get("/games/week", response_model=list[GameOut])
def list_games_for_week(
    start: str, schedule: list[dict] = Depends(get_schedule), db_path: Path = Depends(get_db_path)
) -> list[GameOut]:
    games = get_games_for_week(schedule, start)
    return [_game_out(g, db_path) for g in games]


@router.get("/games/{game_id}", response_model=GameDetailOut)
def get_game_detail(
    game_id: str,
    schedule: list[dict] = Depends(get_schedule),
    db_path: Path = Depends(get_db_path),
    injuries: list[dict] = Depends(get_injury_report_best_effort),
) -> GameDetailOut:
    game = get_game(schedule, game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game: {game_id}")

    markets = [
        MarketPredictionOut(
            market=row["market"], selection=row["selection"], model_probability=row["model_probability"],
            market_probability=row["market_probability"], edge=row["edge"], bookmaker=row["bookmaker"],
            american_odds=row["american_odds"], point=row["point"],
            rebuilt=not made_before_tip(row["created_at"], game),
        )
        for row in store.get_market_predictions_for_game(db_path, game_id)
    ]

    head_to_head = [
        HeadToHeadMeetingOut(
            game_id=g["game_id"], game_date=g["game_date"], home_team=g["home_team"], away_team=g["away_team"],
            home_pts=g.get("home_pts"), away_pts=g.get("away_pts"),
        )
        for g in get_head_to_head(schedule, game["home_team"], game["away_team"], before_date=game["game_date"])
    ]

    base = _game_out(game, db_path)

    # Injury summary: the availability gate's own sentence
    picks = picks_by_player_stat(db_path, game_id, game)
    out_entries = resolve_out_players(injuries, {key[0] for key in picks})
    doubtful_entries = resolve_doubtful_players(injuries, {key[0] for key in picks})
    injury_summary = f"Checked against ESPN availability — {len(out_entries)} Out, {len(doubtful_entries)} Day-to-Day"

    return GameDetailOut(
        **base.model_dump(),
        markets=markets,
        head_to_head=head_to_head,
        home_recent_form=get_recent_form(schedule, game["home_team"], before_date=game["game_date"]),
        away_recent_form=get_recent_form(schedule, game["away_team"], before_date=game["game_date"]),
        injury_summary=injury_summary,
    )


# The MAE is a property of the tracking database and the schedule, not of the request:
# computing it ran two queries per scheduled game (about 3,500 for a full season) on
# EVERY /games/{id}/players call. It changes only when a game resolves, i.e. when the
# database file changes, so it is computed once per (file state, schedule size).
#
# Computed, it is not free: about a second of bulk reads over the resolved rows, so
# charging it to whichever visitor arrives first after the database changed is the
# whole of the cost this cache leaves behind. Two things remove that charge from the
# request path, and both have to hold:
#
#   * the entry is WARMED at startup and re-warmed on a database change
#     (start_mae_warmer / note_database_changed below), so the first request after
#     either already finds the value;
#   * the entry has an explicit age bound, because the file-state key is a
#     best-effort signal and a cache that can never expire is how the availability
#     gate ended up reading one frozen injury report on every request (NBA#18). Past
#     the bound the value is recomputed, and if the recompute FAILS the request gets
#     the failure -- an expired number is never served quietly.
_MAE_CACHE: dict[tuple, tuple[float, dict[str, float | None]]] = {}

# Fifteen minutes, the same bound as the injury report. The database state key
# invalidates this cache far more often than that in practice (every ingest, every
# refresh-odds); the TTL is the backstop for a change the key somehow misses, so it
# is short and never unbounded. Env-overridable like INJURY_CACHE_TTL_SECONDS.
MAE_CACHE_TTL_SECONDS = int(os.getenv("MAE_CACHE_TTL_SECONDS", "900"))

# How often the background warmer checks whether the database moved. Cheap: one
# stat() per tick, and a bulk read only when the state differs from the last warm.
# This is the backstop for a write made outside this process (a mounted volume
# updated elsewhere, a restore); the write paths this process owns re-warm
# themselves via note_database_changed, so the window is only ever this long.
MAE_WARM_POLL_SECONDS = float(os.getenv("MAE_WARM_POLL_SECONDS", "10"))


def _db_state(db_path: Path) -> tuple[int, int]:
    try:
        st = Path(db_path).stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return (-1, -1)


# One bulk read at a time. A visitor arriving while the startup warm is still
# running used to start a SECOND identical read, and the two then serialised on
# the tracking store's global lock -- measurably worse than either alone (2.96 s
# for the first local request against 1.09 s cold and 1.0 s warm, tests/
# test_api_mae_warm.py). Taking this lock and re-checking the cache under it means
# the late request waits for the warm it was about to duplicate and reuses it.
# The wait is bounded by MAE_COMPUTE_WAIT_SECONDS, after which the request
# computes for itself rather than waiting on a hung warm: serving must never
# block indefinitely on the cache.
_MAE_COMPUTE_LOCK = threading.Lock()
MAE_COMPUTE_WAIT_SECONDS = float(os.getenv("MAE_COMPUTE_WAIT_SECONDS", "30"))


def _drop_mae_entries(db_path: Path) -> None:
    """Every entry for one database. One entry per database is kept: an older
    state of the same file can never be asked for again, so holding it would only
    grow the dict -- and holding it past a failed refresh is worse than growing."""
    prefix = str(db_path)
    for stale in [k for k in _MAE_CACHE if k[0] == prefix]:
        del _MAE_CACHE[stale]


def _mae_record(db_path: Path, schedule: list[dict], *, now=time.monotonic) -> dict[str, dict]:
    """The props error estimate per stat, as BOTH figures, in one pass.

    ``{"all": {stat: mae}, "pre_tip": {stat: mae}, "n_all": {...}, "n_pre_tip": {...}}``

    `all` is the headline: every COUNTED pick, one per (game, player, stat) --
    the earliest recorded, whenever it was made (predictor-hub #66,
    2026-10-01). `pre_tip` is the same summariser over the subset whose own
    timestamps prove they were made before tip-off, which is the honest read of
    what the model would have said on the night.

    Both come from ONE `resolved_player_props` call, not two: they are two
    frames of the same rows, filtered by the `made_before_tip` each row
    already carries, so they cannot disagree about which rows exist and the
    per-database-state cache stays worth having.

    ``now`` is a clock, injected so the age bound is testable without sleeping.
    A recompute that fails propagates, and drops the entry on the way out: a
    request must never be handed a value it already knows is past its bound.
    """
    key = (str(db_path), *_db_state(db_path), len(schedule))
    cached = _MAE_CACHE.get(key)
    if cached is not None and now() - cached[0] < MAE_CACHE_TTL_SECONDS:
        return cached[1]

    # Wait for an in-flight compute (the startup warm, most likely) rather than
    # duplicating it, then re-check: the entry that compute produced is the one we
    # wanted. Bounded, so a warm that hangs costs this request its own compute
    # instead of its availability.
    waited = _MAE_COMPUTE_LOCK.acquire(timeout=MAE_COMPUTE_WAIT_SECONDS)
    try:
        key = (str(db_path), *_db_state(db_path), len(schedule))
        cached = _MAE_CACHE.get(key)
        if cached is not None and now() - cached[0] < MAE_CACHE_TTL_SECONDS:
            return cached[1]
        try:
            rows = resolved_player_props(db_path, schedule)
            pre_tip_rows = [r for r in rows if r["made_before_tip"]]
            result = {
                "all": in_sample_mae_by_stat(rows),
                "pre_tip": in_sample_mae_by_stat(pre_tip_rows),
                "n_all": _rows_per_stat(rows),
                "n_pre_tip": _rows_per_stat(pre_tip_rows),
            }
        except Exception:
            _drop_mae_entries(db_path)
            raise
        _drop_mae_entries(db_path)
        _MAE_CACHE[key] = (now(), result)
        return result
    finally:
        if waited:
            _MAE_COMPUTE_LOCK.release()


def _rows_per_stat(rows: list[dict]) -> dict[str, int]:
    """How many resolved rows each stat's MAE was computed from.

    An error estimate with no n beside it is a number nobody can weigh, and the
    pre-tip figure is the one most likely to be thin -- so the n travels with
    both, on the wire and in the provenance sentence the site prints.
    """
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["stat"]] = counts.get(row["stat"], 0) + 1
    return counts


def _mae_by_stat(db_path: Path, schedule: list[dict], *, now=time.monotonic) -> dict[str, float | None]:
    """The HEADLINE in-sample MAE per stat: every counted pick, whenever made.

    Kept under this name and this shape because it is what `PlayerPropOut.mae`
    carries and what every existing caller reads. What changed is the
    population behind it -- see `_mae_record`, which is where both figures and
    their n are built.
    """
    return _mae_record(db_path, schedule, now=now)["all"]


def warm_mae_cache(db_path: Path, schedule: list[dict], *, now=time.monotonic) -> bool:
    """Fill the MAE cache for the current database state, off the request path.

    Returns whether it succeeded and never raises: a warm that fails must not take
    the API down, and must not leave the previous database's number behind looking
    current. It logs the failure instead of swallowing it, because a warm that
    silently never succeeds is a cold path nobody is told about.
    """
    try:
        _mae_by_stat(db_path, schedule, now=now)
    except Exception as exc:  # noqa: BLE001 - reported, not raised: see docstring
        logger.error("MAE cache warm failed for %s: %s: %s", db_path, type(exc).__name__, exc)
        return False
    return True


def note_database_changed(db_path: Path, schedule: list[dict]) -> bool:
    """The write path's own re-warm. run_ingest is what writes predictions and
    outcomes, so it knows the MAE has changed; it says so here instead of waiting
    for the poller to notice. Reuses the existing (mtime, size) invalidation --
    no second notion of "the database changed" is introduced.

    The track-record player-props aggregate is re-warmed here too. Its own bound
    is a TTL rather than a database-state key, so this is a courtesy, not the
    invalidation mechanism: a write that lands between two TTL expiries would
    otherwise leave a five-minute-old aggregate on the page for the rest of its
    window. A warm that fails is reported and ignored here -- the TTL still
    expires and the next request recomputes."""
    warm_player_props_cache(db_path)
    return warm_mae_cache(db_path, schedule)


def start_mae_warmer(
    db_path: Path,
    schedule_path: Path,
    *,
    clock=time.monotonic,
    sleep=time.sleep,
    should_stop=None,
) -> threading.Thread:
    """Warm the MAE cache now, then re-warm it whenever the database changes.

    A daemon thread, started from the app's lifespan: serving begins whether or not
    the warm finishes, and a warm that hangs cannot hold shutdown either. ``clock``,
    ``sleep`` and ``should_stop`` are injected so the loop is testable without
    waiting on a real 30-second tick.

    The invalidation signal is _db_state -- (mtime_ns, size) of the tracking
    database, the key this cache already keys on. An unchanged database costs one
    stat() per tick and nothing else.
    """
    db_path, schedule_path = Path(db_path), Path(schedule_path)

    def _run() -> None:
        last_state: tuple[int, int] | None = None
        while should_stop is None or not should_stop():
            state = _db_state(db_path)
            if state != last_state:
                schedule = load_schedule(schedule_path)
                warm_mae_cache(db_path, schedule, now=clock)
                last_state = state
            sleep(MAE_WARM_POLL_SECONDS)

    thread = threading.Thread(target=_run, name="mae-warmer", daemon=True)
    thread.start()
    return thread


@router.get("/games/{game_id}/players", response_model=list[PlayerPropOut])
def get_game_players(
    game_id: str,
    schedule: list[dict] = Depends(get_schedule),
    db_path: Path = Depends(get_db_path),
    injuries: list[dict] = Depends(get_injury_report),
) -> list[PlayerPropOut]:
    """One projection per player and stat, with an out player left out.

    The out player's rows are ABSENT from this ranking rather than flagged in
    place; they are served once by /games/{game_id}/players/out, attributed
    and dated.
    """
    game = get_game(schedule, game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game: {game_id}")

    name_by_id = load_player_name_map(config.DATA_DIR / "cache" / "hub" / "players.json")
    picks = picks_by_player_stat(db_path, game_id, game)
    actual_by_key = actuals_by_player_stat(db_path, game_id)

    # Resolve availability before ranking anything. Anything unresolvable here
    # removes nobody -- see api/availability.py for why that asymmetry.
    out_ids = {entry["player_id"] for entry in resolve_out_players(injuries, {key[0] for key in picks})}

    mae_record = _mae_record(db_path, schedule)
    mae_by_stat = mae_record["all"]
    mae_pre_tip_by_stat = mae_record["pre_tip"]
    n_by_stat = mae_record["n_all"]
    n_pre_tip_by_stat = mae_record["n_pre_tip"]

    props = []
    for (player_id, stat), (pick, rebuilt) in picks.items():
        if player_id in out_ids:
            continue
        props.append(
            PlayerPropOut(
                player_id=player_id,
                player_name=name_by_id.get(player_id, player_id),
                stat=stat,
                predicted_value=pick["predicted_value"],
                actual_value=actual_by_key.get((player_id, stat)),
                rebuilt=rebuilt,
                mae=mae_by_stat.get(stat),
                mae_pre_tip=mae_pre_tip_by_stat.get(stat),
                mae_n=n_by_stat.get(stat, 0),
                mae_n_pre_tip=n_pre_tip_by_stat.get(stat, 0),
            )
        )
    return props


@router.get("/games/{game_id}/players/out", response_model=list[OutPlayerOut])
def get_game_out_players(
    game_id: str,
    schedule: list[dict] = Depends(get_schedule),
    db_path: Path = Depends(get_db_path),
    injuries: list[dict] = Depends(get_injury_report),
) -> list[OutPlayerOut]:
    """Players removed from this game's ranking by the availability gate.

    One entry per player, whatever their number of stat rows, each carrying
    the feed it came from and the date on that feed.
    """
    game = get_game(schedule, game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game: {game_id}")

    picks = picks_by_player_stat(db_path, game_id, game)
    entries = resolve_out_players(injuries, {key[0] for key in picks})

    # Prefer the roster's own name for the id, the same source the ranking
    # uses, so the out line and the ranking never disagree on who this is.
    name_by_id = load_player_name_map(config.DATA_DIR / "cache" / "hub" / "players.json")
    return [
        OutPlayerOut(**{**entry, "player_name": name_by_id.get(entry["player_id"], entry["player_name"])})
        for entry in entries
    ]


@router.get("/games/{game_id}/players/doubtful", response_model=list[DoubtfulPlayerOut])
def get_game_doubtful_players(
    game_id: str,
    schedule: list[dict] = Depends(get_schedule),
    db_path: Path = Depends(get_db_path),
    injuries: list[dict] = Depends(get_injury_report),
) -> list[DoubtfulPlayerOut]:
    """Players the gate flagged as doubtful, each STILL RANKED.

    A separate feed from /players/out on purpose, not duplication. /out is a
    removal instruction: TopCalls.tsx builds its out-id set from every row it is
    handed and filters the ranking by it. Putting doubtful players there would
    make the shipped frontend delete all 52 of them, which is the defect this
    feed exists to close rather than a rendering of it. The frontend joins this
    feed to the ranking on player_id -- an exact key, for the same reason the
    gate is -- and renders a subordinate note beside the pick. Nobody is removed.

    Like the out feed it depends on the real availability dependency, so a feed
    we cannot read refuses with 503 instead of serving an empty list. An empty
    list here means "the report was read and names nobody doubtful", which is a
    different fact from "nobody checked" and must never look alike.
    """
    game = get_game(schedule, game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game: {game_id}")

    picks = picks_by_player_stat(db_path, game_id, game)
    entries = resolve_doubtful_players(injuries, {key[0] for key in picks})

    # The roster's own name for the id, the same source the ranking uses, so
    # the note and the row it attaches to never disagree on who this is.
    name_by_id = load_player_name_map(config.DATA_DIR / "cache" / "hub" / "players.json")
    return [
        DoubtfulPlayerOut(**{**entry, "player_name": name_by_id.get(entry["player_id"], entry["player_name"])})
        for entry in entries
    ]


@router.get("/hub/teams")
def hub_teams() -> list[dict]:
    return load_hub_cache(config.DATA_DIR / "cache" / "hub" / "teams.json")


@router.get("/hub/players")
def hub_players() -> list[dict]:
    return load_hub_cache(config.DATA_DIR / "cache" / "hub" / "players.json")


@router.get("/hub/rankings")
def hub_rankings() -> list[dict]:
    return load_hub_cache(config.DATA_DIR / "cache" / "hub" / "rankings.json")


@router.get("/hub/standings")
def hub_standings() -> list[dict]:
    return load_hub_cache(config.DATA_DIR / "cache" / "hub" / "standings.json")


@router.get("/hub/track-record", response_model=list[TrackRecordOut])
def hub_track_record(
    db_path: Path = Depends(get_db_path), schedule: list[dict] = Depends(get_schedule)
) -> list[TrackRecordOut]:
    return compute_track_record(db_path, schedule)


@router.get("/hub/vs-market", response_model=VsMarketOut)
def hub_vs_market(
    db_path: Path = Depends(get_db_path), schedule: list[dict] = Depends(get_schedule)
) -> VsMarketOut:
    """Model against the price on the moneyline: headline, weekly rows over
    the same window as /hub/track-record, the scope reconciliation, and the
    method sentences the page prints verbatim."""
    return compute_vs_market(db_path, schedule)


@router.get("/value-picks")
def value_picks(
    db_path: Path = Depends(get_db_path),
    max_age_minutes: int = 60,
    threshold: float = edge_gate.EDGE_THRESHOLD,
    ceiling: float = edge_gate.EDGE_CEILING,
) -> dict:
    """The gated picks: at most one single per game, 5% edge, fresh odds only.

    A measurement surface, not a recommendation. `picks` is frequently empty and
    that is a real answer -- a stale slate or a day with no edge both produce
    zero, and the thresholds are returned alongside so a reader can see what was
    applied instead of taking it on trust.

    The disclaimer is PL's framing, deliberately: a positive yield on a graded
    ledger is not evidence that the gate found an edge. Nothing here places a
    bet.
    """
    # `.isoformat()`, not a datetime object. `created_at` is TEXT holding an
    # ISO-8601 string, and SQLite compares it as text -- but sqlite3 renders a
    # datetime argument with a SPACE separator ("2026-10-06 13:16:09+00:00")
    # while these rows use "T". " " sorts before "T", so every row then looked
    # newer than `since` and the freshness filter was silently a no-op: a
    # two-hour-old line would sail through as fresh.
    since = (datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)).isoformat()
    rows = [dict(r) for r in get_recent_market_predictions(db_path, since=since)]

    # Count suspect rows: those that are otherwise eligible (fresh, have line, not parlay)
    # but have edge > ceiling
    suspect_rows = []
    for row in rows:
        if row.get("market") == "parlay":
            continue
        point = row.get("point")
        if row.get("market") in ("spread", "total") and point is None:
            continue
        edge = row.get("edge")
        if edge is None:
            continue
        if not _is_fresh(row, datetime.now(timezone.utc)):
            continue
        if not edge_gate._has_counterpart(row, rows):
            continue
        if edge > ceiling:
            suspect_rows.append(row)

    picks = edge_gate.gated_picks(rows, threshold=threshold, ceiling=ceiling)
    return {
        "picks": picks,
        "edge_threshold": threshold,
        "edge_ceiling": ceiling,
        "n_suspect": len(suspect_rows),
        "max_odds_age_minutes": max_age_minutes,
        "n_odds_rows_considered": len(rows),
        "disclaimer": (
            "A selection threshold, not advice. Historical hit rate and yield on "
            "this ledger are measurements of past games, not evidence of a "
            "profitable strategy, and nothing here places a bet."
        ),
    }


@router.get("/snapshot-meta")
def get_snapshot_meta() -> dict:
    """When the data this site serves was last written, as an ISO-8601 UTC string.

    NBA is the odd one out: the routes above read the tracking DB, the schedule
    cache, and the hub caches, never public_snapshot.json, so there is no
    generated_at to report verbatim. The honest equivalent is the newest mtime
    across exactly the files those routes actually read -- the tracking DB
    (predictions, market rows, outcomes), the schedule cache (games.json, read
    by nearly every route including /games/week, which is what the hub's NBA
    teaser calls), and data/cache/hub/*.json (teams/players/rankings/
    standings, plus the player-name map) -- which is when the numbers the
    visitor is looking at were produced. Reporting the snapshot file's
    timestamp instead would describe numbers nobody is served, so that was
    ruled out deliberately; do not "fix" this to read the snapshot.

    "source" keeps the vocabulary shared with the other four sport APIs
    ("public_snapshot" when files exist, "live" otherwise) because the hub
    switches on it -- even though here the value is a filesystem mtime, not a
    snapshot stamp. Every path missing means "nothing written yet", reported
    as source "live" with a null timestamp rather than raising: a public
    deploy before its first refresh is a real state, not a failure. The
    exists() guard also covers a TRACKING_DB_PATH (env-overridable) pointing
    at a path with no local file.
    """
    # Built from config.DATA_DIR, not config.CACHE_DIR, deliberately: every
    # hub route above spells the path as config.DATA_DIR / "cache" / "hub" /
    # <name>.json, and CACHE_DIR is frozen at import from the original
    # DATA_DIR, so only the DATA_DIR form keeps this set exactly the files
    # those routes read when DATA_DIR is redirected (tests, PROJECT_ROOT env).
    # Identical to CACHE_DIR / "hub" in any normal deployment. The schedule
    # cache comes from the get_schedule_path() helper itself -- the same
    # function get_schedule() depends on -- rather than a re-spelled literal,
    # so the two cannot drift. data/cache/training/games.json is deliberately
    # excluded: only the admin retrain POST reads it (guarded on exists()),
    # no public GET serves it, so it is not part of "these numbers".
    hub_dir = config.DATA_DIR / "cache" / "hub"
    paths = [
        config.TRACKING_DB_PATH,
        get_schedule_path(),
        *sorted(hub_dir.glob("*.json")),
    ]
    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    if not mtimes:
        return {"generated_at": None, "source": "live"}
    newest = max(mtimes)
    return {
        "generated_at": datetime.fromtimestamp(newest, tz=timezone.utc).isoformat(),
        "source": "public_snapshot",
    }


@router.get("/calibration")
def calibration(
    db_path: Path = Depends(get_db_path), schedule: list[dict] = Depends(get_schedule)
) -> list[dict]:
    return compute_model_calibration(db_path, schedule)


@router.get("/manifest")
def get_manifest(models_dir: Path = Depends(get_models_dir)) -> dict:
    manifest_path = models_dir / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="No manifest found — run /retrain first")
    return json.loads(manifest_path.read_text())


@router.get("/player-props-manifest")
def get_player_props_manifest(models_dir: Path = Depends(get_models_dir)) -> dict:
    manifest_path = models_dir / "player_props_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="No player props manifest found — run /retrain first")
    return json.loads(manifest_path.read_text())


@router.post("/retrain", dependencies=[Depends(require_admin)])
def retrain(
    models_dir: Path = Depends(get_models_dir),
    training_games_path: Path = Depends(get_training_games_path),
) -> dict:
    if not training_games_path.exists():
        raise HTTPException(status_code=400, detail="No training games cache found")

    games = pd.DataFrame(json.loads(training_games_path.read_text()))
    model_version = datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")
    trained_at = datetime.now(timezone.utc).isoformat()

    return run_retrain_pipeline(games, models_dir, model_version=model_version, trained_at=trained_at)


def _market_stds_from_manifest(models_dir: Path) -> tuple[float, float]:
    margin_std, total_std = 12.0, 15.0
    manifest_path = models_dir / "manifest.json"
    if manifest_path.exists():
        metrics = json.loads(manifest_path.read_text()).get("metrics", {})
        margin_mae = metrics.get("margin", {}).get("mae")
        total_mae = metrics.get("total", {}).get("mae")
        if margin_mae is not None:
            margin_std = std_from_mae(margin_mae)
        if total_mae is not None:
            total_std = std_from_mae(total_mae)
    return margin_std, total_std


@router.post("/refresh-odds", dependencies=[Depends(require_admin)], status_code=202)
def refresh_odds(
    schedule: list[dict] = Depends(get_schedule),
    db_path: Path = Depends(get_db_path),
    models_dir: Path = Depends(get_models_dir),
) -> dict:
    margin_std, total_std = _market_stds_from_manifest(models_dir)
    stored = refresh_market_predictions(schedule, db_path, margin_std=margin_std, total_std=total_std)
    # This wrote to the tracking database, which is what invalidates the cached
    # MAE -- even though these market rows are not what the MAE is computed from.
    # Re-warm here so the next visitor is not charged for a change we made.
    note_database_changed(db_path, schedule)
    return {"status": "ok", "market_predictions_stored": stored}


@router.get("/season/first-week", response_model=SeasonBoundsOut)
def season_first_week(schedule: list[dict] = Depends(get_schedule), today: str = Depends(get_today)) -> SeasonBoundsOut:
    return SeasonBoundsOut(first_week_start=default_week_start(schedule, today))


_ingest_status: dict = {"running": False, "last_result": None, "last_error": None}


def _run_ingest_background(db_path: Path, schedule_path: Path, models_dir: Path) -> None:
    _ingest_status["running"] = True
    _ingest_status["last_error"] = None
    try:
        summary = run_ingest(
            "2025-10-01", "2026-11-30", player_hub_days=9999, db_path=db_path, log=lambda _msg: None
        )
        # Team win/margin/total predictions are scored by run_ingest above;
        # the odds-derived markets (spread/total lines, bookmaker, edge)
        # need a separate pass against the sportsbook API, keyed off the
        # schedule and model MAE that ingest just refreshed.
        schedule = load_schedule(schedule_path)
        margin_std, total_std = _market_stds_from_manifest(models_dir)
        summary["market_predictions_stored"] = refresh_market_predictions(
            schedule, db_path, margin_std=margin_std, total_std=total_std
        )
        # Both passes above wrote to the tracking database, which is what
        # invalidates the cached MAE, so this is the moment to replace it rather
        # than leave the next visitor to pay for it. warm_mae_cache never raises,
        # so this cannot turn a successful ingest into a failed one.
        note_database_changed(db_path, schedule)
        _ingest_status["last_result"] = summary
    except Exception as exc:  # noqa: BLE001 - reported via status endpoint, not re-raised (background task)
        _ingest_status["last_error"] = str(exc)
    finally:
        _ingest_status["running"] = False


@router.post("/admin/refresh-full", dependencies=[Depends(require_admin)], status_code=202)
def refresh_full(
    background_tasks: BackgroundTasks,
    db_path: Path = Depends(get_db_path),
    schedule_path: Path = Depends(get_schedule_path),
    models_dir: Path = Depends(get_models_dir),
) -> dict:
    """Runs the full real-data pipeline (schedule/box-score fetch, model
    retraining, and scoring — including upcoming games and player props)
    against this server's own live tracking DB. For a deployment with no
    baked-in predictions and no interactive shell access — the same thing
    `python -m nba_predictor.pipeline.ingest` does locally, triggered over
    HTTP instead. Runs in the background; poll GET /admin/refresh-full/status
    for progress, since a full run can take a long time."""
    if _ingest_status["running"]:
        return {"status": "already_running"}
    background_tasks.add_task(_run_ingest_background, db_path, schedule_path, models_dir)
    return {"status": "started"}


@router.get("/admin/refresh-full/status", dependencies=[Depends(require_admin)])
def refresh_full_status() -> dict:
    return _ingest_status
