import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from nba_predictor.api.deps import (
    get_db_path,
    get_models_dir,
    get_schedule,
    get_today,
    get_training_games_path,
    require_admin,
)
from nba_predictor.api.schemas import (
    GameDetailOut,
    GameOut,
    HeadToHeadMeetingOut,
    MarketPredictionOut,
    PlayerPropOut,
    PredictionOut,
    SeasonBoundsOut,
    TeamOut,
    TrackRecordOut,
)
from nba_predictor.data.team_reference import TEAMS, get_team
from nba_predictor.odds.value_bets import std_from_mae
from nba_predictor.pipeline.ingest import run_ingest
from nba_predictor.pipeline.refresh_odds import refresh_market_predictions
from nba_predictor.pipeline.retrain import run_retrain_pipeline
from nba_predictor.services.calibration_service import compute_model_calibration
from nba_predictor.services.hub_service import compute_track_record, load_hub_cache, load_player_name_map
from nba_predictor.services.schedule_repository import (
    default_week_start,
    get_game,
    get_games_for_date,
    get_games_for_week,
    get_head_to_head,
    get_recent_form,
)
from nba_predictor.tracking import store
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


def _prediction_out(db_path: Path, game_id: str) -> PredictionOut | None:
    row = store.get_latest_prediction_for_game(db_path, game_id)
    if row is None:
        return None
    return PredictionOut(
        home_win_probability=row["home_win_prob"],
        predicted_margin=row["predicted_margin"],
        predicted_total=row["predicted_total"],
    )


def _game_out(g: dict, db_path: Path) -> GameOut:
    return GameOut(
        game_id=g["game_id"], game_date=g["game_date"], home_team=g["home_team"], away_team=g["away_team"],
        prediction=_prediction_out(db_path, g["game_id"]),
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


@router.get("/games/{game_id}/players", response_model=list[PlayerPropOut])
def get_game_players(
    game_id: str, schedule: list[dict] = Depends(get_schedule), db_path: Path = Depends(get_db_path)
) -> list[PlayerPropOut]:
    game = get_game(schedule, game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game: {game_id}")

    name_by_id = load_player_name_map(config.DATA_DIR / "cache" / "hub" / "players.json")
    outcomes = store.get_player_outcomes_for_game(db_path, game_id)
    actual_by_key = {(row["player_id"], row["stat"]): row["actual_value"] for row in outcomes}

    return [
        PlayerPropOut(
            player_id=row["player_id"],
            player_name=name_by_id.get(row["player_id"], row["player_id"]),
            stat=row["stat"],
            predicted_value=row["predicted_value"],
            actual_value=actual_by_key.get((row["player_id"], row["stat"])),
        )
        for row in store.get_player_predictions_for_game(db_path, game_id)
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


@router.post("/refresh-odds", dependencies=[Depends(require_admin)], status_code=202)
def refresh_odds(
    schedule: list[dict] = Depends(get_schedule),
    db_path: Path = Depends(get_db_path),
    models_dir: Path = Depends(get_models_dir),
) -> dict:
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

    stored = refresh_market_predictions(schedule, db_path, margin_std=margin_std, total_std=total_std)
    return {"status": "ok", "market_predictions_stored": stored}


@router.get("/season/first-week", response_model=SeasonBoundsOut)
def season_first_week(schedule: list[dict] = Depends(get_schedule), today: str = Depends(get_today)) -> SeasonBoundsOut:
    return SeasonBoundsOut(first_week_start=default_week_start(schedule, today))


_ingest_status: dict = {"running": False, "last_result": None, "last_error": None}


def _run_ingest_background(db_path: Path) -> None:
    _ingest_status["running"] = True
    _ingest_status["last_error"] = None
    try:
        summary = run_ingest(
            "2025-10-01", "2026-11-30", player_hub_days=9999, db_path=db_path, log=lambda _msg: None
        )
        _ingest_status["last_result"] = summary
    except Exception as exc:  # noqa: BLE001 - reported via status endpoint, not re-raised (background task)
        _ingest_status["last_error"] = str(exc)
    finally:
        _ingest_status["running"] = False


@router.post("/admin/refresh-full", dependencies=[Depends(require_admin)], status_code=202)
def refresh_full(background_tasks: BackgroundTasks, db_path: Path = Depends(get_db_path)) -> dict:
    """Runs the full real-data pipeline (schedule/box-score fetch, model
    retraining, and scoring — including upcoming games and player props)
    against this server's own live tracking DB. For a deployment with no
    baked-in predictions and no interactive shell access — the same thing
    `python -m nba_predictor.pipeline.ingest` does locally, triggered over
    HTTP instead. Runs in the background; poll GET /admin/refresh-full/status
    for progress, since a full run can take a long time."""
    if _ingest_status["running"]:
        return {"status": "already_running"}
    background_tasks.add_task(_run_ingest_background, db_path)
    return {"status": "started"}


@router.get("/admin/refresh-full/status", dependencies=[Depends(require_admin)])
def refresh_full_status() -> dict:
    return _ingest_status
