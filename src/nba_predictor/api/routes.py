import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from nba_predictor.api.deps import (
    get_db_path,
    get_models_dir,
    get_schedule,
    get_training_games_path,
    require_admin,
)
from nba_predictor.api.schemas import (
    GameDetailOut,
    GameOut,
    MarketPredictionOut,
    PlayerPropOut,
    PredictionOut,
    TeamOut,
    TrackRecordOut,
)
from nba_predictor.data.team_reference import TEAMS, get_team
from nba_predictor.pipeline.refresh_odds import refresh_market_predictions
from nba_predictor.pipeline.retrain import run_retrain_pipeline
from nba_predictor.services.hub_service import compute_track_record, load_hub_cache
from nba_predictor.services.schedule_repository import get_game, get_games_for_date, get_games_for_week
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
            american_odds=row["american_odds"],
        )
        for row in store.get_market_predictions_for_game(db_path, game_id)
    ]

    base = _game_out(game, db_path)
    return GameDetailOut(**base.model_dump(), markets=markets)


@router.get("/games/{game_id}/players", response_model=list[PlayerPropOut])
def get_game_players(
    game_id: str, schedule: list[dict] = Depends(get_schedule), db_path: Path = Depends(get_db_path)
) -> list[PlayerPropOut]:
    game = get_game(schedule, game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game: {game_id}")

    return [
        PlayerPropOut(
            player_id=row["player_id"], player_name=row["player_id"], stat=row["stat"],
            predicted_value=row["predicted_value"],
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
def refresh_odds(schedule: list[dict] = Depends(get_schedule), db_path: Path = Depends(get_db_path)) -> dict:
    stored = refresh_market_predictions(schedule, db_path)
    return {"status": "ok", "market_predictions_stored": stored}
