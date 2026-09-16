import json
from datetime import datetime, timezone
from pathlib import Path

from nba_predictor import config
from nba_predictor.services.hub_service import compute_track_record, load_hub_cache
from nba_predictor.services.schedule_repository import load_schedule
from nba_predictor.tracking.store import init_db


def generate_snapshot(schedule_path: Path, hub_dir: Path, manifest_path: Path, db_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schedule": load_schedule(schedule_path),
        "hub": {
            "teams": load_hub_cache(hub_dir / "teams.json"),
            "players": load_hub_cache(hub_dir / "players.json"),
            "rankings": load_hub_cache(hub_dir / "rankings.json"),
            "standings": load_hub_cache(hub_dir / "standings.json"),
        },
        "track_record": [record.model_dump() for record in compute_track_record(db_path)],
        "manifest": manifest,
    }


def write_snapshot(snapshot: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2))


def main() -> None:
    init_db(config.TRACKING_DB_PATH)
    snapshot = generate_snapshot(
        schedule_path=config.DATA_DIR / "cache" / "schedule" / "games.json",
        hub_dir=config.DATA_DIR / "cache" / "hub",
        manifest_path=config.PROJECT_ROOT / "models" / "manifest.json",
        db_path=config.TRACKING_DB_PATH,
    )
    write_snapshot(snapshot, config.DATA_DIR / "public_snapshot.json")


if __name__ == "__main__":
    main()
