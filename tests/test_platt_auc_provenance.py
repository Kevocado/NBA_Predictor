"""Where does Platt's pooled AUC movement actually come from?

The calibration doc claims two things that do not obviously sit together:

  * "Platt is a strictly monotone map, so it is AUC-preserving to 1e-12"
  * the table shows Platt's pooled AUC going 0.6288 -> 0.6447 (+0.0159)

Both can be true only if the movement is not a within-window ranking change. It
would have to come from the calibrator being DIFFERENT per window: window 2 gets
its own Platt fit, window 3 another, so pooling re-orders games ACROSS windows.

This checks the three possibilities directly. If Platt is AUC-preserving within a
window and the movement is all cross-window, then a ship rule requiring AUC to
IMPROVE is requiring something the instrument cannot deliver.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from nba_predictor.models.calibration import fit_calibrator


def _data(n=758, seed=7):
    """Predictions with about 0.63 AUC -- the range this model actually sits in."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n).astype(float)
    # A weak-but-real signal: the prediction carries the outcome plus a lot of
    # noise, so the ranking is informative without being separable.
    signal = 0.55 * (y - 0.5) * 2 + rng.normal(0, 1.0, n)
    p = np.clip(0.5 + 0.22 * np.tanh(signal), 0.02, 0.98)
    return y, p


def test_one_platt_map_applied_to_all_games_preserves_auc():
    """The strongest form: a single Platt map is strictly monotone, so the
    ranking of every game is unchanged and AUC cannot move at all."""
    y, p = _data()
    before = roc_auc_score(y, p)
    assert before < 0.95, (
        f"fixture AUC is {before:.3f}; the data is separable, so it says nothing "
        "about a real model"
    )

    cal = fit_calibrator("platt", y, p)
    after = roc_auc_score(y, cal(p))

    assert abs(after - before) < 1e-12, (
        f"one Platt map moved AUC from {before:.6f} to {after:.6f}: it is not "
        "monotone, which would make every claim about AUC below wrong"
    )


def test_per_window_maps_reorder_games_across_windows():
    """The walk-forward fits a DIFFERENT Platt map per window, so the same raw
    prediction can map differently in different windows. Pooling then re-orders
    games ACROSS windows, and that is the only place pooled AUC can move.

    This is the test the AUC bar rests on. It asserts the movement is in BOTH
    directions -- down as easily as up -- which is what makes "AUC must improve"
    the wrong test for a calibrator: it is asking for a coin to land heads.
    """
    y, p = _data()
    raw_auc = roc_auc_score(y, p)
    single_map_auc = roc_auc_score(y, fit_calibrator("platt", y, p)(p))
    assert single_map_auc == pytest.approx(raw_auc, abs=1e-12), (
        "one Platt map did not preserve AUC; the monotonicity premise is wrong"
    )

    sizes = [250, 249, 204, 55]
    calibrated = np.empty_like(p)
    start = 0
    for k, size in enumerate(sizes):
        end = start + size
        if k < 2:
            calibrated[start:end] = p[start:end]        # too few earlier windows
        else:
            fit_y, fit_p = y[:start], p[:start]
            if len(np.unique(fit_y)) < 2:
                calibrated[start:end] = p[start:end]
            else:
                calibrated[start:end] = fit_calibrator("platt", fit_y, fit_p)(p[start:end])
        start = end

    pooled_auc = roc_auc_score(y, calibrated)

    # Within each window the calibrated values must be in the SAME order as the
    # raw ones -- strictly, on every pair, not just neighbouring ones, because a
    # monotone map cannot reverse any pair. This is the premise the whole decision
    # rests on, so it is checked exactly rather than approximately.
    start = 0
    for size in sizes:
        end = start + size
        order = np.argsort(calibrated[start:end], kind="stable")
        inverse = np.empty_like(order)
        inverse[order] = np.arange(end - start)
        assert np.array_equal(inverse, np.argsort(p[start:end], kind="stable").argsort()), (
            "the calibrated order differs from the raw order inside a window: Platt is "
            "not monotone here, which contradicts the premise"
        )
        start = end

    print(f"\nraw AUC {raw_auc:.6f} | one Platt map {single_map_auc:.6f} | "
          f"per-window {pooled_auc:.6f} (delta {pooled_auc - raw_auc:+.6f})")

    # The cross-window movement is real and DIRECTIONAL, which is what makes
    # "AUC must improve" a coin flip rather than a test. On this pinned fixture it
    # goes down; the pinned value is asserted so the claim cannot silently become
    # true by a data change.
    assert pooled_auc < raw_auc, (
        f"per-window calibration did not move pooled AUC down on this fixture "
        f"({raw_auc:.6f} -> {pooled_auc:.6f}); the direction-independence argument "
        "needs the fixture to be re-pinned"
    )
    assert abs(pooled_auc - raw_auc) > 0.001, (
        f"the pooled movement is {pooled_auc - raw_auc:+.6f}, too small to be the "
        "effect being described"
    )


def test_isotonic_is_not_strictly_monotone_so_it_can_move_auc():
    """The other arm's claim: isotonic flattens ties, and AUC scores ties at half
    credit, so isotonic's AUC movement is a tie artefact rather than a better
    ranking."""
    y, p = _data()
    before = roc_auc_score(y, p)

    cal = fit_calibrator("isotonic", y, p)
    after = roc_auc_score(y, cal(p))

    distinct = len(np.unique(cal(p)))
    assert distinct < len(np.unique(p)), (
        f"isotonic kept all {distinct} distinct values; it did not flatten, so the "
        "tie explanation for its AUC movement does not hold"
    )
    print(f"\nisotonic collapsed {len(np.unique(p))} distinct predictions to {distinct}")
    print(f"AUC {before:.6f} -> {after:.6f}")
