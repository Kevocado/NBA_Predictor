import json
import sqlite3
from pathlib import Path

from nba_predictor.api.schemas import ConfidenceBucketOut, PropStatOut, TrackRecordOut
from nba_predictor.tracking import store
from nba_predictor.tracking.store import get_connection


_CONFIDENCE_BUCKETS = [("50-60%", 0.50, 0.60), ("60-70%", 0.60, 0.70), ("70%+", 0.70, 1.01)]


def _bucket_for(prob: float) -> str | None:
    for label, lo, hi in _CONFIDENCE_BUCKETS:
        if lo <= prob < hi:
            return label
    return None


def load_hub_cache(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def load_player_name_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    return {row["player_id"]: row["player_name"] for row in rows}


def _settle_game_outcome(db_path: Path, schedule: list[dict]) -> TrackRecordOut | None:
    """Settles the model's own win/loss call (predictions.home_win_prob >= 0.5)
    against each game's actual result. This is fully real: every prediction
    was computed from real pre-game rolling features and every result here
    is a real completed game, joined via the schedule cache's home_pts/
    away_pts (populated by pipeline/ingest.py)."""
    schedule_by_id = {g["game_id"]: g for g in schedule}
    predictions = store.get_all_predictions(db_path)

    total = 0
    correct = 0
    for prediction in predictions:
        game = schedule_by_id.get(prediction["game_id"])
        if game is None or not game.get("completed") or game.get("home_pts") is None:
            continue
        total += 1
        predicted_home_win = prediction["home_win_prob"] >= 0.5
        actual_home_win = game["home_pts"] > game["away_pts"]
        if predicted_home_win == actual_home_win:
            correct += 1

    if total == 0:
        return None
    return TrackRecordOut(
        market="game_outcome", total_predictions=total, correct_predictions=correct, hit_rate=round(correct / total, 3)
    )


def _settle_h2h_market_predictions(db_path: Path, schedule: list[dict]) -> TrackRecordOut | None:
    """Settles game_market_predictions rows for market="h2h" (selection is a
    team abbreviation) against actual results. Spread/total market rows
    aren't settled here — the stored row has no point/line value, so
    "did it cover" isn't computable from what's tracked; those markets
    still show total_predictions with hit_rate 0.0 rather than a fabricated
    result (see the market-count fallback below)."""
    schedule_by_id = {g["game_id"]: g for g in schedule}
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM game_market_predictions WHERE market = 'h2h'").fetchall()

    total = 0
    correct = 0
    bucket_stats: dict[str, list[int]] = {label: [0, 0] for label, _, _ in _CONFIDENCE_BUCKETS}
    for row in rows:
        game = schedule_by_id.get(row["game_id"])
        if game is None or not game.get("completed") or game.get("home_pts") is None:
            continue
        total += 1
        actual_winner = game["home_team"] if game["home_pts"] > game["away_pts"] else game["away_team"]
        hit = row["selection"] == actual_winner
        if hit:
            correct += 1
        bucket = _bucket_for(row["model_probability"])
        if bucket is not None:
            bucket_stats[bucket][0] += 1
            if hit:
                bucket_stats[bucket][1] += 1

    if total == 0:
        return None
    buckets = [
        ConfidenceBucketOut(
            bucket=label,
            total_predictions=b_total,
            correct_predictions=b_correct,
            hit_rate=round(b_correct / b_total, 3) if b_total else 0.0,
        )
        for label, (b_total, b_correct) in bucket_stats.items()
    ]
    return TrackRecordOut(market="h2h", total_predictions=total, correct_predictions=correct,
                          hit_rate=round(correct / total, 3), confidence_buckets=buckets)


def _summarize_player_props(db_path: Path) -> TrackRecordOut | None:
    """Joins prediction snapshots to recorded outcomes. Uses the latest
    snapshot per (game_id, player_id, stat) so re-scores don't double-count.
    mean_signed_error is mean(predicted - actual): positive means systematic
    over-prediction. Snapshots with no recorded position group as "Unknown"."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT s.game_id, s.player_id, s.stat, s.predicted_value,
                   o.actual_value, s.position, s.created_at
            FROM player_prediction_snapshots s
            JOIN game_player_outcomes o
              ON o.game_id = s.game_id AND o.player_id = s.player_id AND o.stat = s.stat
            """
        ).fetchall()
    if not rows:
        return None
    latest: dict[tuple[str, str, str], sqlite3.Row] = {}
    for r in rows:
        key = (r["game_id"], r["player_id"], r["stat"])
        if key not in latest or r["created_at"] > latest[key]["created_at"]:
            latest[key] = r
    by_stat: dict[str, list[float]] = {}
    by_pos: dict[str, list[float]] = {}
    for r in latest.values():
        err = r["predicted_value"] - r["actual_value"]
        by_stat.setdefault(r["stat"], []).append(err)
        by_pos.setdefault(r["position"] or "Unknown", []).append(abs(err))
    per_stat = [
        PropStatOut(stat=stat, n=len(errs),
                    mae=round(sum(abs(e) for e in errs) / len(errs), 3),
                    mean_signed_error=round(sum(errs) / len(errs), 3))
        for stat, errs in sorted(by_stat.items())
    ]
    return TrackRecordOut(
        market="player_props",
        total_predictions=len(latest),
        correct_predictions=0,
        hit_rate=0.0,
        per_stat=per_stat,
        per_position_mae={p: round(sum(v) / len(v), 3) for p, v in sorted(by_pos.items())},
    )


def compute_track_record(db_path: Path, schedule: list[dict] | None = None) -> list[TrackRecordOut]:
    schedule = schedule or []
    rows: list[TrackRecordOut] = []

    with get_connection(db_path) as conn:
        market_counts = conn.execute(
            "SELECT market, COUNT(*) as total FROM game_market_predictions GROUP BY market"
        ).fetchall()

    settled_h2h = _settle_h2h_market_predictions(db_path, schedule)
    for row in market_counts:
        if row["market"] == "h2h" and settled_h2h is not None:
            rows.append(settled_h2h)
        else:
            rows.append(TrackRecordOut(market=row["market"], total_predictions=row["total"], correct_predictions=0, hit_rate=0.0))

    game_outcome = _settle_game_outcome(db_path, schedule)
    if game_outcome is not None:
        rows.append(game_outcome)

    player_props = _summarize_player_props(db_path)
    if player_props is not None:
        rows.append(player_props)

    return rows
