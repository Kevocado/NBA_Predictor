import json
from pathlib import Path

from nba_predictor.api.schemas import TrackRecordOut
from nba_predictor.tracking.store import get_connection


def load_hub_cache(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def compute_track_record(db_path: Path) -> list[TrackRecordOut]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT market, COUNT(*) as total FROM game_market_predictions GROUP BY market"
        ).fetchall()

    return [
        TrackRecordOut(market=row["market"], total_predictions=row["total"], correct_predictions=0, hit_rate=0.0)
        for row in rows
    ]
