"""The offline artifact gate (spec section 3.5, ported from the NFL repo).

Prop models in this repo shipped on in-sample metrics — numbers a model cannot
fail, because it was fitted on the rows it was scored against. The gate exists
to refuse an artifact whose held-out evidence does not clear a calibration
check, which is the only way in-sample evaluation gets caught at all.

`MAX_GAP` / `MIN_N` are the NFL pattern: ±5pt per bucket, at least 100
predictions per bucket so a bucket is not asserting calibration from a handful
of games. **Both are named constants because Kevin decides them** — they are not
tuned to let a particular model through, and `tests/test_artifact_gate.py` pins
the Phase A measurement against them so loosening either one fails a test rather
than quietly reversing a conclusion.

Two distinct refusals, because they call for different fixes:

* **inconclusive** (`n < MIN_N`) — too little evidence to judge. Refusing is the
  honest response; a gate that waves through "not enough data" is
  indistinguishable from no gate.
* **miscalibrated** (`gap > MAX_GAP`) — confidently wrong, refuse.

The gate is deliberately boring: no scoring model, no thresholds that adapt to
the data. It answers one question — is there enough held-out evidence, and does
it agree with what the model said?
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

#: Maximum allowed |observed − predicted| in any prediction bucket.
MAX_GAP = 0.05

#: Minimum predictions required before the calibration check means anything.
MIN_N = 100


def bucket_calibration(y_true, p_pred, n_buckets: int = 5) -> list[dict]:
    """Observed vs predicted rate per bucket of predicted probability.

    Buckets are equal-count quantiles of the predictions, so each holds the same
    number of games and a small bucket cannot assert calibration from two rows.
    Equal-count rather than equal-width on purpose: NBA win probabilities cluster
    around 0.55, and fixed-width buckets would put almost everything in one.

    The buckets partition the predictions exactly — `sum(n) == len(y)`.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"y_true and p_pred must align: {y.shape} vs {p.shape}")

    frame = pd.DataFrame({"p": p, "y": y})
    # rank(method="first") breaks ties so identical predictions cannot land in
    # one bucket twice and leave another empty.
    frame["bucket"] = pd.qcut(frame["p"].rank(method="first"), n_buckets, labels=False)

    table = []
    for bucket, group in frame.groupby("bucket"):
        predicted = float(group["p"].mean())
        observed = float(group["y"].mean())
        table.append(
            {
                "bucket": int(bucket),
                "predicted": predicted,
                "observed": observed,
                "gap": observed - predicted,
                "n": int(len(group)),
            }
        )
    table.sort(key=lambda row: row["predicted"])
    return table


def calibration_gap(y_true, p_pred, n_buckets: int = 5) -> float:
    """Worst absolute gap across buckets. NaN when there is nothing to judge."""
    table = bucket_calibration(y_true, p_pred, n_buckets)
    if not table:
        return float("nan")
    return max(abs(row["gap"]) for row in table)


def gate_artifact(metrics: dict, max_gap: float = MAX_GAP, min_n: int = MIN_N) -> bool:
    """True only if the held-out evidence clears calibration and is big enough.

    Expects `{"calibration_max_gap": float, "n": int}`. Anything missing,
    non-numeric, NaN or non-positive is a refusal — an inconclusive gate is not
    a passing gate, and `nan <= max_gap` being False while
    `not (nan > max_gap)` is True is precisely the trap that would let a broken
    fit through.
    """
    if not isinstance(metrics, dict):
        return False

    gap, n = metrics.get("calibration_max_gap"), metrics.get("n")
    if gap is None or n is None:
        return False
    try:
        gap = float(gap)
        n = int(n)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(gap):
        return False

    return gap <= max_gap and n >= min_n


def gate_predictions(y_true, p_pred, min_n: int = MIN_N, n_buckets: int = 5) -> tuple[bool, dict]:
    """Convenience: measure held-out predictions and gate on the result.

    Returns `(passed, evidence)` where evidence carries the per-bucket table so
    the refusal is explainable rather than a bare False — "refused" without the
    buckets that refused it is the kind of thing nobody trusts next season.
    """
    table = bucket_calibration(y_true, p_pred, n_buckets)
    n = int(len(np.asarray(y_true)))
    gap = calibration_gap(y_true, p_pred, n_buckets)
    evidence = {
        "calibration_max_gap": gap,
        "n": n,
        "buckets": table,
        "max_gap_allowed": MAX_GAP,
        "min_n_required": min_n,
    }
    return gate_artifact({"calibration_max_gap": gap, "n": n}, min_n=min_n), evidence