"""The offline artifact gate: refuse to write a model that cannot clear a
calibration check on held-out data (spec section 3.5, NFL pattern).

This exists because the repo shipped prop models whose only evidence was
in-sample metrics -- numbers a model cannot fail. A gate that only runs
in-sample cannot catch that; it needs held-out predictions and an honest
reference to compare them against.

Two failures it must distinguish, because they need different fixes:

* **too few predictions to judge** (n < `MIN_N`) -- the check is inconclusive,
  so refuse rather than wave it through. An inconclusive gate that passes is
  indistinguishable from no gate.
* **too much error where it matters** (max bucket gap > `MAX_GAP`) -- the model
  is confidently wrong, so refuse.

Measured on the real Phase A winner: max bucket gap 0.0592 at n=758, against a
0.05 bar. **That model is refused by this gate**, which is the honest outcome and
the reason it is worth having.
"""

import numpy as np
import pytest

from nba_predictor.models.artifact_gate import (
    MAX_GAP,
    MIN_N,
    bucket_calibration,
    calibration_gap,
    gate_artifact,
)


def test_gate_refuses_an_uncalibrated_model():
    assert gate_artifact({"calibration_max_gap": 0.09, "n": 500}) is False
    assert gate_artifact({"calibration_max_gap": 0.03, "n": 500}) is True


def test_gate_refuses_when_there_is_too_little_data_to_judge():
    """n too small is not a pass. A well-calibrated number computed from 40
    predictions is not evidence of calibration, and admitting it would let a
    model through on noise."""
    assert gate_artifact({"calibration_max_gap": 0.01, "n": MIN_N - 1}) is False
    assert gate_artifact({"calibration_max_gap": 0.01, "n": MIN_N}) is True


def test_gate_refuses_metrics_with_nothing_to_judge_at_all():
    """No data is not a pass. A model with no holdout predictions has produced
    no evidence, and `n=0` must not slip under a `>=`."""
    for metrics in ({}, {"calibration_max_gap": 0.0}, {"n": 500}, {"calibration_max_gap": None, "n": 500}):
        assert gate_artifact(metrics) is False, f"{metrics} should not pass the gate"


def test_a_nan_gap_does_not_pass_because_nan_comparisons_are_false():
    """`nan <= 0.05` is False, which would make this fail -- but `not (nan > x)`
    is True, which would make it PASS. Guard the comparison explicitly, because
    a broken fit that returns NaN is exactly what this gate exists to catch."""
    assert gate_artifact({"calibration_max_gap": float("nan"), "n": 500}) is False


def test_calibration_gap_is_the_worst_bucket_disagreement():
    rng = np.random.default_rng(0)

    # An honestly calibrated model: draw a probability, then sample the outcome
    # from it. (Building it the other way -- from the label -- produces a model
    # that is confidently wrong, which is the case the next assertion covers.)
    p = np.clip(rng.beta(2, 2, 20000), 0.01, 0.99)
    y = (rng.random(20000) < p).astype(int)
    assert calibration_gap(y, p, n_buckets=5) < 0.03, "a calibrated model must clear the bar"

    # And a confidently wrong one must not: predicts 0.8/0.99 for a 50/50 event.
    skewed = np.where(y == 1, 0.99, 0.80)
    assert calibration_gap(y, skewed, n_buckets=5) > 0.05


def test_bucket_calibration_reports_observed_against_predicted():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4000)
    p = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.15, 4000), 0.01, 0.99)

    table = bucket_calibration(y, p, n_buckets=5)
    assert len(table) == 5
    for row in table:
        assert 0.0 <= row["predicted"] <= 1.0
        assert 0.0 <= row["observed"] <= 1.0
        assert row["n"] > 0, "a bucket with no predictions was reported"
        assert row["gap"] == pytest.approx(row["observed"] - row["predicted"])


def test_buckets_are_ordered_and_cannot_overlap():
    """Buckets must partition the predictions. Overlapping buckets would let a
    miscalibrated model satisfy the bar by appearing well-predicted twice."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 4000)
    p = np.clip(rng.random(4000) * 0.6 + 0.2, 0.01, 0.99)

    table = bucket_calibration(y, p, n_buckets=5)
    predicted = [row["predicted"] for row in table]
    assert predicted == sorted(predicted), f"buckets out of order: {predicted}"
    assert sum(row["n"] for row in table) == len(y), "buckets did not partition the predictions"


def test_the_phase_a_win_model_is_refused_by_this_gate():
    """The number from docs/nba-parity-evaluation-2026-10.md, pinned. The race
    winner scored a 0.0592 maximum bucket gap at n=758 against a 0.05 bar, so
    this gate refuses it. If someone later tunes the constant until this passes,
    this test is what catches that."""
    assert gate_artifact({"calibration_max_gap": 0.0592, "n": 758}) is False
    assert gate_artifact({"calibration_max_gap": 0.0592, "n": 758}) == (
        0.0592 <= MAX_GAP and 758 >= MIN_N
    ), "the pinned measurement no longer agrees with the constants"


def test_gate_accepts_the_incumbents_real_headroom_when_it_improves():
    assert gate_artifact({"calibration_max_gap": MAX_GAP, "n": MIN_N}) is True