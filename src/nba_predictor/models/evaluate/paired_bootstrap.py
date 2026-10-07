"""Paired bootstrap: is a difference real, or is it noise on 758 games?

A point estimate moving from 0.6636 to 0.6542 says almost nothing on its own.
On 758 games a difference that size happens by chance routinely, and reporting
the improvement as a result without an interval is how a model ships for free.
So every metric here is only called improved when its 95% interval **excludes
zero**.

**Paired, because the two models are scored on the same games.** Each resample
draws game indices once and applies them to both models, so the game-to-game
difficulty cancels. An unpaired bootstrap would inflate the interval with
between-game variance that both models already agree about, and make a real
improvement look like noise.

Direction is per-metric: lower is better for log-loss, Brier and MAE, higher for
AUC. `improved` encodes that, so a caller cannot accidentally read a worse AUC as
a win.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

#: metric name -> (callable(y, score) -> float, lower_is_better)
METRICS: dict[str, tuple] = {
    "log_loss": (lambda y, p: float(log_loss(y, p, labels=[0, 1])), True),
    "brier": (lambda y, p: float(np.mean((p - y) ** 2)), True),
    "auc": (lambda y, p: float(roc_auc_score(y, p)), False),
    "mae": (lambda y, p: float(np.mean(np.abs(p - y))), True),
}

DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 20261007


def paired_bootstrap(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    metric: str,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap the metric difference `candidate - baseline` on the same games.

    Returns the two point estimates, their difference, the 95% interval of that
    difference, and `improved`: True only when the interval excludes zero **in
    the direction that counts as better**.

    Resamples in which the metric is undefined (an AUC draw that happens to
    contain one class only) are dropped rather than scored as 0.0, because a
    fabricated 0.5 would drag the interval toward "no difference" for a reason
    that has nothing to do with the models. The count actually used is
    returned as `n_resamples_used`.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(METRICS)}")
    fn, lower_is_better = METRICS[metric]

    y = np.asarray(y, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if not (y.shape == baseline.shape == candidate.shape):
        raise ValueError(
            f"y, baseline and candidate must align: {y.shape}, {baseline.shape}, {candidate.shape}"
        )
    n = len(y)
    if n == 0:
        raise ValueError("nothing to bootstrap")
    if metric in ("log_loss", "brier", "auc"):
        # Probabilities must be strictly inside (0, 1) for log-loss.
        baseline = np.clip(baseline, 1e-12, 1 - 1e-12)
        candidate = np.clip(candidate, 1e-12, 1 - 1e-12)

    base_point = fn(y, baseline)
    cand_point = fn(y, candidate)

    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)  # ONE draw, applied to both models
        yb = y[idx]
        try:
            d = fn(yb, candidate[idx]) - fn(yb, baseline[idx])
        except ValueError:
            # One class in this draw (AUC only); a number here would be invented.
            continue
        if not np.isfinite(d):
            continue
        diffs.append(d)

    diffs_arr = np.asarray(diffs)
    if not len(diffs_arr):
        raise ValueError("no resample produced a defined metric; the data is degenerate")

    lo, hi = (np.percentile(diffs_arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])).tolist()
    excludes_zero = (lo > 0) or (hi < 0)
    improved = (hi < 0) if lower_is_better else (lo > 0)

    return {
        "metric": metric,
        "lower_is_better": lower_is_better,
        "baseline": base_point,
        "candidate": cand_point,
        "difference": cand_point - base_point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "alpha": alpha,
        "n": int(n),
        "n_resamples": int(n_resamples),
        "n_resamples_used": int(len(diffs_arr)),
        "seed": seed,
        "excludes_zero": bool(excludes_zero),
        "improved": bool(improved and excludes_zero),
    }


def bootstrap_table(
    y: np.ndarray,
    baseline_scores: dict[str, np.ndarray],
    candidate_scores: dict[str, np.ndarray],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict]:
    """Run every metric in `baseline_scores`, paired, with one shared seed."""
    return {
        name: paired_bootstrap(
            y, baseline_scores[name], candidate_scores[name], name,
            n_resamples=n_resamples, seed=seed,
        )
        for name in baseline_scores
    }
