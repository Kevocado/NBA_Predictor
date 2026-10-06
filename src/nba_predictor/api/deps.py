from datetime import date
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


def get_injury_report() -> list[dict]:
    """ESPN's live injury report, rows keyed by the ESPN athlete id.

    A dependency rather than a direct call, so the availability gate's feed is
    substitutable and no test can reach the network through it.

    A feed we cannot read is NOT a report that came back empty, and the two
    must not look alike: the props route turns this into a 503 rather than
    serving a ranking nobody checked for availability. Decision 6 of the
    fixture-insight spec is that NBA picks do not ship without the gate.
    """
    from nba_predictor.data import espn

    try:
        return espn.get_injuries()
    except Exception as exc:  # noqa: BLE001 - any feed failure fails closed
        raise HTTPException(
            status_code=503,
            detail=(
                "Availability could not be checked: ESPN's injury report is "
                f"unavailable ({type(exc).__name__}). Player picks are withheld "
                "rather than ranked without an availability gate."
            ),
        ) from exc


def get_injury_report_best_effort() -> list[dict]:
    """Best-effort injury report for game detail: returns empty list on failure.

    The game detail page must render even when ESPN is unavailable. The
    injury_summary will reflect the unavailable state. Player-pick routes
    continue to use the fail-closed get_injury_report dependency.
    """
    from nba_predictor.data import espn

    try:
        return espn.get_injuries()
    except Exception:
        return []


def require_admin() -> None:
    if config.PUBLIC_MODE:
        raise HTTPException(status_code=404, detail="Not found")


def get_models_dir() -> Path:
    return config.PROJECT_ROOT / "models"


def get_training_games_path() -> Path:
    return config.DATA_DIR / "cache" / "training" / "games.json"


def get_today() -> str:
    return date.today().isoformat()
