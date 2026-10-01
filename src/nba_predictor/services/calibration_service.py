from pathlib import Path

import numpy as np

from nba_predictor.models.evaluate.calibration import compute_calibration_bins
from nba_predictor.services.hub_service import pre_tip_picks


def compute_model_calibration(db_path: Path, schedule: list[dict], n_bins: int = 10) -> list[dict]:
    """Reliability-curve bins for the model's stored win-probability calls,
    settled against real completed games from the schedule cache.

    **Unchanged by the 2026-10-01 track-record reversal, deliberately.** A
    bucket here is not a track record, it is a price: the hub compares a live
    probability against these buckets to decide how much to trust the model, so
    a bucket fed with picks made after tip-off would be telling the hub a
    probability is better calibrated than it is, on evidence that could not have
    existed when the price was quoted. So the pre-tip filter stays exactly as
    strict here, and only the track record was reversed. If this ever should
    follow, that is a pricing change with its own review, not a side effect.

    The population is therefore `pre_tip_picks` -- the pre-tip SUBSET of the
    counted picks, which is the secondary figure the track record publishes
    beside its headline. The two views cannot describe different games, because
    the subset is taken from the same counted set rather than re-derived by a
    second rule."""
    picks = pre_tip_picks(db_path, schedule)
    y_true = [1 if game["home_pts"] > game["away_pts"] else 0 for game, _ in picks]
    y_prob = [pick["home_win_prob"] for _, pick in picks]

    if not y_true:
        return []

    return compute_calibration_bins(np.array(y_true), np.array(y_prob), n_bins=n_bins)
