import numpy as np

from nba_predictor.models.evaluate.paired_bootstrap import paired_gap_bootstrap


def _data(n, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.9, n)
    y = (rng.uniform(size=n) < p).astype(int)
    return y, p


def test_an_identical_candidate_is_not_widened_and_the_interval_is_centred_on_zero():
    y, p = _data(1500)
    out = paired_gap_bootstrap(y, p, p, n_resamples=300)
    assert out["difference"] == 0 and out["ci_low"] <= 0 <= out["ci_high"] and not out["widened"]


def test_a_reliably_miscalibrated_candidate_is_widened():
    y, p = _data(3000, seed=1)
    bad = np.clip(p + 0.25, 0.01, 0.99)          # systematically over-confident
    out = paired_gap_bootstrap(y, p, bad, n_resamples=300)
    assert out["widened"] and out["ci_low"] > 0


def test_small_samples_give_wide_intervals_so_a_point_wobble_is_not_called_real():
    y, p = _data(120, seed=2)
    wobble = np.clip(p + np.random.default_rng(3).normal(0, 0.03, len(p)), 0.01, 0.99)
    assert not paired_gap_bootstrap(y, p, wobble, n_resamples=300)["widened"]
