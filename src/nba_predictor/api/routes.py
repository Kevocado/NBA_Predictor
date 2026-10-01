import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from nba_predictor.api.availability import resolve_out_players
from nba_predictor.api.deps import (
    get_db_path,
    get_injury_report,
    get_models_dir,
    get_schedule,
    get_schedule_path,
    get_today,
    get_training_games_path,
    require_admin,
)
from nba_predictor.api.schemas import (
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
from nba_predictor.pipeline.refresh_odds import refresh_market_predictions
from nba_predictor.pipeline.retrain import run_retrain_pipeline
from nba_predictor.services.calibration_service import compute_model_calibration
from nba_predictor.services.hub_service import (
    compute_track_record,
    compute_vs_market,
    load_hub_cache,
    load_player_name_map,
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
from nba_predictor.tracking.player_props import (
    actuals_by_player_stat,
    picks_by_player_stat,
    resolved_player_props,
)
from nba_predictor.tracking.timing import latest_by_instant, latest_pre_tip, made_before_tip
from nba_predictor import config

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


def _prediction_out(db_path: Path, game: dict) -> tuple[PredictionOut | None, bool]:
    """The pick to show and whether it was rebuilt after tip-off. The latest
    pick made before tip-off wins; a later backtest row only shows (labelled
    rebuilt) when no pre-tip pick exists."""
    rows = store.get_predictions_for_game(db_path, game["game_id"])
    if not rows:
        return None, False
    row = latest_pre_tip(rows, game)
    rebuilt = row is None
    row = row if row is not None else latest_by_instant(rows)
    return (
        PredictionOut(
            home_win_probability=row["home_win_prob"],
            predicted_margin=row["predicted_margin"],
            predicted_total=row["predicted_total"],
        ),
        rebuilt,
    )


def _game_out(g: dict, db_path: Path) -> GameOut:
    prediction, rebuilt = _prediction_out(db_path, g)
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
    game_id: str, schedule: list[dict] = Depends(get_schedule), db_path: Path = Depends(get_db_path)
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
    return GameDetailOut(
        **base.model_dump(),
        markets=markets,
        head_to_head=head_to_head,
        home_recent_form=get_recent_form(schedule, game["home_team"], before_date=game["game_date"]),
        away_recent_form=get_recent_form(schedule, game["away_team"], before_date=game["game_date"]),
    )


def _mae_by_stat(db_path: Path, schedule: list[dict]) -> dict[str, float | None]:
    """In-sample MAE per stat over resolved rows only. None when a stat has
    been resolved never -- see models.player_props.in_sample_mae_by_stat."""
    return in_sample_mae_by_stat(resolved_player_props(db_path, schedule))


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

    mae_by_stat = _mae_by_stat(db_path, schedule)

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
