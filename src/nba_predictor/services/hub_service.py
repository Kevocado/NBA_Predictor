import json
from pathlib import Path

from nba_predictor.api.schemas import TrackRecordOut
from nba_predictor.tracking import store
from nba_predictor.tracking.store import get_connection
from nba_predictor.tracking.timing import latest_pre_tip, made_before_tip


def load_hub_cache(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def load_player_name_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    return {row["player_id"]: row["player_name"] for row in rows}


def pre_tip_picks(db_path: Path, schedule: list[dict]) -> tuple[list[tuple[dict, dict]], int]:
    """(game, latest pick made before tip-off) for every completed game, plus
    how many completed games have only picks rebuilt after tip-off. The
    backtest writes completed games into the same table long after they were
    played; those rows are never judged."""
    schedule_by_id = {g["game_id"]: g for g in schedule}
    by_game: dict[str, list] = {}
    with get_connection(db_path) as conn:
        for row in conn.execute("SELECT * FROM predictions"):
            by_game.setdefault(row["game_id"], []).append(row)

    picks: list[tuple[dict, dict]] = []
    rebuilt = 0
    for game_id, rows in by_game.items():
        game = schedule_by_id.get(game_id)
        if game is None or not game.get("completed") or game.get("home_pts") is None:
            continue
        pick = latest_pre_tip(rows, game)
        if pick is None:
            rebuilt += 1
        else:
            picks.append((game, pick))
    return picks, rebuilt


def _settle_game_outcome(db_path: Path, schedule: list[dict]) -> TrackRecordOut | None:
    """Settles the model's own win/loss call (predictions.home_win_prob >= 0.5)
    against each game's actual result, using only the pick made before
    tip-off. Every result is a real completed game, joined via the schedule
    cache's home_pts/away_pts (populated by pipeline/ingest.py)."""
    picks, rebuilt = pre_tip_picks(db_path, schedule)
    if not picks and not rebuilt:
        return None

    correct = sum(
        1 for game, pick in picks
        if (pick["home_win_prob"] >= 0.5) == (game["home_pts"] > game["away_pts"])
    )
    total = len(picks)
    return TrackRecordOut(
        market="game_outcome", total_predictions=total, correct_predictions=correct,
        hit_rate=round(correct / total, 3) if total else 0.0, n_rebuilt=rebuilt,
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

    completed_rows = 0
    total = 0
    correct = 0
    for row in rows:
        game = schedule_by_id.get(row["game_id"])
        if game is None or not game.get("completed") or game.get("home_pts") is None:
            continue
        completed_rows += 1
        if not made_before_tip(row["created_at"], game):
            continue
        total += 1
        actual_winner = game["home_team"] if game["home_pts"] > game["away_pts"] else game["away_team"]
        if row["selection"] == actual_winner:
            correct += 1

    if completed_rows == 0:
        return None
    return TrackRecordOut(
        market="h2h", total_predictions=total, correct_predictions=correct,
        hit_rate=round(correct / total, 3) if total else 0.0,
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

    return rows
