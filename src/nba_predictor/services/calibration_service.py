from pathlib import Path

import numpy as np

from nba_predictor.models.evaluate.calibration import compute_calibration_bins
from nba_predictor.services.hub_service import pre_tip_picks


def compute_model_calibration(db_path: Path, schedule: list[dict], n_bins: int = 10) -> list[dict]:
    """Reliability-curve bins for the model's stored win-probability calls,
    settled against real completed games from the schedule cache — the
    same pre-tip (prediction, actual-result) pairs hub_service's
    game_outcome track record uses; picks rebuilt after tip-off are left out."""
    picks, _ = pre_tip_picks(db_path, schedule)
    y_true = [1 if game["home_pts"] > game["away_pts"] else 0 for game, _ in picks]
    y_prob = [pick["home_win_prob"] for _, pick in picks]

    if not y_true:
        return []

    return compute_calibration_bins(np.array(y_true), np.array(y_prob), n_bins=n_bins)
