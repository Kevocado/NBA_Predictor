import numpy as np
import pandas as pd
import pytest


def test_chronological_split_holdout_is_most_recent():
    from nba_predictor.models.evaluate.walk_forward import chronological_split

    df = pd.DataFrame(
        {
            "game_date": pd.date_range("2026-10-01", periods=10).astype(str),
            "value": range(10),
        }
    )
    train, holdout = chronological_split(df, date_col="game_date", holdout_fraction=0.2)

    assert len(train) == 8
    assert len(holdout) == 2
    assert train["game_date"].max() < holdout["game_date"].min()


def test_chronological_split_handles_unsorted_input():
    from nba_predictor.models.evaluate.walk_forward import chronological_split

    df = pd.DataFrame(
        {
            "game_date": ["2026-11-03", "2026-11-01", "2026-11-02"],
            "value": [3, 1, 2],
        }
    )
    train, holdout = chronological_split(df, date_col="game_date", holdout_fraction=0.34)

    assert holdout["value"].tolist() == [3]


def test_chronological_split_raises_on_too_few_rows():
    from nba_predictor.models.evaluate.walk_forward import chronological_split

    df = pd.DataFrame({"game_date": ["2026-11-01"], "value": [1]})
    with pytest.raises(ValueError):
        chronological_split(df, date_col="game_date")


def test_compute_calibration_bins_perfect_calibration():
    from nba_predictor.models.evaluate.calibration import compute_calibration_bins

    y_prob = np.array([0.05] * 20 + [0.95] * 20)
    y_true = np.array([0] * 19 + [1] + [1] * 19 + [0])

    bins = compute_calibration_bins(y_true, y_prob, n_bins=10)

    assert len(bins) == 2
    low_bin = next(b for b in bins if b["bin_start"] < 0.5)
    high_bin = next(b for b in bins if b["bin_start"] >= 0.5)
    assert low_bin["count"] == 20
    assert high_bin["count"] == 20
    assert low_bin["actual_rate"] == pytest.approx(0.05)
    assert high_bin["actual_rate"] == pytest.approx(0.95)


def test_compute_calibration_bins_omits_empty_bins():
    from nba_predictor.models.evaluate.calibration import compute_calibration_bins

    y_prob = np.array([0.5, 0.5])
    y_true = np.array([1, 0])

    bins = compute_calibration_bins(y_true, y_prob, n_bins=10)
    assert len(bins) == 1
