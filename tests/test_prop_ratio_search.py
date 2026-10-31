"""Helpers of the pre-registered ratio search: Holm, the one-sided paired t, and the season label."""
import math

import numpy as np
import pandas as pd
import pytest

from nba_predictor.models.evaluate.prop_ratio_search import holm, one_sided_t_p, season_label


def test_holm_matches_hand_computation():
    # Sorted: 0.01, 0.04, 0.03 -> adjusted 0.03, 0.06, 0.06 in sorted order, then monotone.
    adj = holm([0.04, 0.01, 0.03])
    assert adj[1] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06)
    assert adj[0] == pytest.approx(0.06)


def test_holm_is_capped_at_one():
    assert max(holm([0.9, 0.95, 0.99])) == 1.0


def test_one_sided_t_agrees_with_numpy():
    rng = np.random.default_rng(1)
    d = rng.normal(0.2, 1.0, 500)
    mean, p = one_sided_t_p(float(d.sum()), float((d * d).sum()), len(d))
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
    assert mean == pytest.approx(d.mean())
    assert p == pytest.approx(0.5 * math.erfc(t / math.sqrt(2)))
    assert p < 0.05


def test_season_label_is_october_start_year():
    dates = pd.Series(["2018-10-01", "2019-04-10", "2020-10-02", "2024-01-15", "2024-10-22"])
    assert season_label(dates).tolist() == [2018, 2018, 2020, 2023, 2024]
