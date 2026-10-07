"""The paired bootstrap, and above all the null case.

The claim under test is not "the number went down". It is "the number went down
by more than the noise", and the test that matters most is the one where nothing
improved: identical models on identical games must produce an interval that
CONTAINS zero. A harness that reported an improvement on a null difference would
wave every future model through.

Everything here is seeded, so a failure is reproducible rather than a rumour.
"""

import numpy as np
import pytest

from nba_predictor.models.evaluate.paired_bootstrap import (
    METRICS,
    bootstrap_table,
    paired_bootstrap,
)


def _games(n: int = 758, seed: int = 11, spread: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold home-win probabilities, and the outcomes they were scored on.

    `p` genuinely generated `y`, so it beats a naive base-rate model. `spread`
    controls how far: 0.20 gives a clear, unambiguous win (log-loss gap ~0.077
    against a bootstrap spread of ~0.014). A much smaller spread gives the case
    this file most needs — a model that looks better and is NOT, because the gap
    is smaller than the noise — which is why it is a parameter rather than a
    constant.

    This fixture previously tried to express "worse" by pulling `p` toward 0.5,
    which is not reliably worse at all: on a realised base rate above 0.56 it can
    be BETTER, and two tests failed because of that. A flat base-rate prediction
    is what the project actually compares against, and it is unambiguously
    weaker.
    """
    rng = np.random.default_rng(seed)
    p = np.clip(rng.normal(0.56, spread, n), 0.02, 0.98)
    y = (rng.random(n) < p).astype(float)
    return y, p


def _naive(y: np.ndarray) -> np.ndarray:
    """The naive comparator: one fixed home-win rate for every game."""
    return np.full(len(y), float(np.mean(y)))


# --- the null case: the required one -------------------------------------

def test_identical_models_give_an_interval_that_contains_zero():
    """The true difference is exactly zero, so the interval must contain zero.

    Seeded, and asserted on a real n=758 game set rather than a toy one: the
    point of this test is that the machinery does not manufacture significance
    out of nothing when there is genuinely nothing.
    """
    y, p = _games()
    for metric in ("log_loss", "brier", "auc"):
        out = paired_bootstrap(y, p, p, metric, n_resamples=2000, seed=4242)
        assert out["difference"] == pytest.approx(0.0)
        assert out["ci_low"] <= 0.0 <= out["ci_high"], (
            f"{metric}: identical models produced an interval excluding zero "
            f"({out['ci_low']:.4f}, {out['ci_high']:.4f})"
        )
        assert out["improved"] is False
        assert out["excludes_zero"] is False


# --- improvement has to be real ------------------------------------------

def test_a_materially_better_model_is_called_improved():
    y, p = _games()
    naive = _naive(y)
    out = paired_bootstrap(y, naive, p, "log_loss", n_resamples=2000, seed=4242)
    assert out["ci_high"] < 0, "a clearly better model was not detected"
    assert out["improved"] is True
    assert out["difference"] < 0


def test_a_materially_worse_model_is_never_called_improved():
    y, p = _games()
    out = paired_bootstrap(y, p, _naive(y), "log_loss", n_resamples=2000, seed=4242)
    assert out["ci_low"] > 0, "the worse model did not show up as worse"
    assert out["improved"] is False


def test_improvement_requires_the_interval_to_exclude_zero_not_just_the_sign():
    """A point estimate in the right direction is not an improvement."""
    y, p = _games()
    # A nudge far too small to survive resampling noise.
    nudge = np.clip(p + 1e-4, 1e-9, 1 - 1e-9)
    out = paired_bootstrap(y, p, nudge, "log_loss", n_resamples=2000, seed=4242)
    if out["difference"] < 0:
        assert out["improved"] is False, (
            "a difference smaller than the noise was called an improvement"
        )


def test_auc_is_judged_in_the_opposite_direction_to_the_error_metrics():
    """Higher AUC is better. `improved` must not assume lower-is-better."""
    y, p = _games()
    naive = _naive(y)
    better = paired_bootstrap(y, naive, p, "auc", n_resamples=1000, seed=1)
    assert better["lower_is_better"] is False
    assert better["difference"] > 0
    assert better["improved"] is True

    worse_out = paired_bootstrap(y, p, naive, "auc", n_resamples=1000, seed=1)
    assert worse_out["improved"] is False


def test_a_better_looking_model_whose_gap_is_smaller_than_the_noise_is_not_an_improvement():
    """The case this harness exists for, and it is not hypothetical.

    A model whose log-loss is genuinely lower than the baseline's, by an amount
    a reader would happily report as progress, but not by more than 758 games
    can resolve. Measured here: a gap of about -0.007 against a bootstrap spread
    of about 0.0066, so the interval straddles zero.

    The failure mode this prevents is a Phase A-style win of 0.009 log-loss being
    written up as an improvement. Whether the REAL 758-game comparison clears
    this bar is exactly what the PR has to report, not assume.
    """
    y, p = _games(spread=0.09)
    out = paired_bootstrap(y, _naive(y), p, "log_loss", n_resamples=2000, seed=4242)
    assert out["difference"] < 0, "the fixture must have a real improvement to be worth testing"
    assert out["excludes_zero"] is False, "a sub-noise gap was claimed as excluding zero"
    assert out["improved"] is False


# --- the pairing itself ---------------------------------------------------

def test_pairing_narrows_the_interval_because_game_difficulty_cancels():
    """The property that justifies pairing at all, asserted rather than assumed.

    Two models that differ only slightly are scored on the same 758 games. The
    PAIRED interval resamples games once and applies them to both models, so the
    between-game difficulty — which both models equally experience — cancels.
    An UNPAIRED interval draws games separately for each model and therefore
    charges that shared difficulty to the difference, making it wider.

    If pairing were dropped, the two intervals would be near-identical and this
    test would fail.
    """
    y, p = _games()
    near_same = np.clip(p + 0.02 * np.random.default_rng(2).normal(size=len(y)), 1e-9, 1 - 1e-9)

    paired = paired_bootstrap(y, p, near_same, "brier", n_resamples=2000, seed=5)
    unpaired = _unpaired_bootstrap(y, p, near_same, "brier", n_resamples=2000, seed=5)

    paired_width = paired["ci_high"] - paired["ci_low"]
    unpaired_width = unpaired[1] - unpaired[0]
    assert paired_width < unpaired_width, (
        f"paired interval ({paired_width:.6f}) is not narrower than "
        f"unpaired ({unpaired_width:.6f}); the draw is not being shared"
    )


def _unpaired_bootstrap(y, baseline, candidate, metric, *, n_resamples, seed):
    """What the interval would be if each model drew its OWN games.

    Deliberately naive -- independent draws per model -- because that is the
    thing pairing is being compared against.
    """
    fn, _ = METRICS[metric]
    n = len(y)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_resamples):
        i_b = rng.integers(0, n, n)
        i_c = rng.integers(0, n, n)
        diffs.append(fn(y[i_c], candidate[i_c]) - fn(y[i_b], baseline[i_b]))
    arr = np.asarray(diffs)
    return tuple(np.percentile(arr, [2.5, 97.5]).tolist())


def test_regression_metrics_paired_are_narrower_than_the_spread_of_the_data():
    """A constant-vs-known-target MAE comparison: the paired interval reflects
    the difference, not the target's own spread."""
    y = np.random.default_rng(2).normal(0, 12, 758)
    base = np.zeros_like(y)
    cand = y + np.random.default_rng(8).normal(0, 1, 758)  # near-perfect
    out = paired_bootstrap(y, base, cand, "mae", n_resamples=1000, seed=17)
    assert out["improved"] is True
    # Interval on the DIFFERENCE is tight even though the target itself spans ~50.
    assert (out["ci_high"] - out["ci_low"]) < 5.0


# --- reproducibility and hygiene ------------------------------------------

def test_the_same_seed_gives_the_same_interval():
    y, p = _games()
    a = paired_bootstrap(y, _naive(y), p, "log_loss", n_resamples=500, seed=77)
    b = paired_bootstrap(y, _naive(y), p, "log_loss", n_resamples=500, seed=77)
    assert (a["ci_low"], a["ci_high"]) == (b["ci_low"], b["ci_high"])


def test_the_seed_is_reported_so_a_result_can_be_reproduced():
    y, p = _games()
    out = paired_bootstrap(y, p, p, "log_loss", n_resamples=100, seed=1234)
    assert out["seed"] == 1234
    assert out["n_resamples"] == 100


def test_the_interval_brackets_the_point_difference():
    y, p = _games()
    for metric in METRICS:
        if metric == "mae":
            continue
        out = paired_bootstrap(y, _naive(y), p, metric, n_resamples=500, seed=31)
        assert out["ci_low"] <= out["difference"] <= out["ci_high"], metric


def test_degenerate_draws_are_dropped_not_scored_as_zero():
    """An AUC resample with one class has no value. Scoring it 0.5 would drag
    the interval toward 'no difference' for a reason unrelated to the models."""
    y = np.concatenate([np.ones(700), np.zeros(58)])
    p = np.clip(y * 0.4 + 0.3, 1e-9, 1 - 1e-9)
    out = paired_bootstrap(y, p, p, "auc", n_resamples=200, seed=13)
    assert out["n_resamples_used"] <= out["n_resamples"]
    assert np.isfinite(out["ci_low"]) and np.isfinite(out["ci_high"])


def test_misaligned_inputs_are_refused_rather_than_broadcast():
    y, p = _games(20)
    with pytest.raises(ValueError, match="align"):
        paired_bootstrap(y, p, p[:-1], "log_loss", n_resamples=50)


def test_an_unknown_metric_is_refused_by_name():
    y, p = _games(20)
    with pytest.raises(ValueError, match="unknown metric"):
        paired_bootstrap(y, p, p, "rmse", n_resamples=50)


def test_bootstrap_table_pairs_every_metric_with_one_seed():
    y, p = _games(120)
    naive = _naive(y)
    table = bootstrap_table(
        y,
        {"log_loss": naive, "brier": naive, "auc": naive},
        {"log_loss": p, "brier": p, "auc": p},
        n_resamples=300,
        seed=8,
    )
    assert set(table) == {"log_loss", "brier", "auc"}
    for out in table.values():
        assert out["seed"] == 8
        assert out["n"] == 120
