import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_SUBDIRS = ["nba_api", "balldontlie", "odds", "sportsbook", "injuries", "espn"]

TRACKING_DB_PATH = Path(os.environ.get("TRACKING_DB_PATH", str(DATA_DIR / "tracking.db")))
_backup = os.environ.get("TRACKING_DB_BACKUP_PATH")
TRACKING_DB_BACKUP_PATH = Path(_backup) if _backup else None

PUBLIC_MODE = os.environ.get("PUBLIC_MODE", "false").lower() == "true"
PUBLIC_SNAPSHOT_POLL_SECONDS = int(os.environ.get("PUBLIC_SNAPSHOT_POLL_SECONDS", "300"))

BALLDONTLIE_API_KEY = os.environ.get("BALLDONTLIE_API_KEY")
SPORTSBOOK_API_KEY = os.environ.get("SPORTSBOOK_API_KEY")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY")


def ensure_cache_dirs() -> None:
    for name in CACHE_SUBDIRS:
        (CACHE_DIR / name).mkdir(parents=True, exist_ok=True)
