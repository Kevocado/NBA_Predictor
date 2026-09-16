from pathlib import Path

from fastapi import Depends, HTTPException

from nba_predictor import config
from nba_predictor.services.schedule_repository import load_schedule


def get_db_path() -> Path:
    return config.TRACKING_DB_PATH


def get_schedule_path() -> Path:
    return config.DATA_DIR / "cache" / "schedule" / "games.json"


def get_schedule(schedule_path: Path = Depends(get_schedule_path)) -> list[dict]:
    return load_schedule(schedule_path)


def require_admin() -> None:
    if config.PUBLIC_MODE:
        raise HTTPException(status_code=404, detail="Not found")


def get_models_dir() -> Path:
    return config.PROJECT_ROOT / "models"


def get_training_games_path() -> Path:
    return config.DATA_DIR / "cache" / "training" / "games.json"
