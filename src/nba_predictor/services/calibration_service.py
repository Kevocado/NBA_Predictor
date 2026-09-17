from pathlib import Path

import numpy as np

from nba_predictor.models.evaluate.calibration import compute_calibration_bins
from nba_predictor.tracking import store


def compute_model_calibration(db_path: Path, schedule: list[dict], n_bins: int = 10) -> list[dict]:
    """Reliability-curve bins for the model's stored win-probability calls,
    settled against real completed games from the schedule cache — the
    same real (prediction, actual-result) pairs hub_service's
    game_outcome track record uses."""
    schedule_by_id = {g["game_id"]: g for g in schedule}
    predictions = store.get_all_predictions(db_path)

    y_true, y_prob = [], []
    for prediction in predictions:
        game = schedule_by_id.get(prediction["game_id"])
        if game is None or not game.get("completed") or game.get("home_pts") is None:
            continue
        y_true.append(1 if game["home_pts"] > game["away_pts"] else 0)
        y_prob.append(prediction["home_win_prob"])

    if not y_true:
        return []

    return compute_calibration_bins(np.array(y_true), np.array(y_prob), n_bins=n_bins)
