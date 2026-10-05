"""One residual-σ probability model, shared by every market (spec section 3.2, §6).

Win probability used to come straight out of `predict_proba`, and cover/over
from independent regressors with no shared error distribution. Nothing tied the
three numbers a visitor reads together: a game could be 60% to win, favoured to
cover by one margin model and under by another, with no inconsistency anyone
could point at.

So there is one fitted Normal residual σ per output, taken from pooled
walk-forward residuals, and every probability is a CDF of it:

    win   = norm.cdf(predicted_margin / sigma_margin)
    cover = norm.cdf((predicted_margin - line) / sigma_margin)
    over  = norm.cdf((predicted_total - line) / sigma_total)

Coherence is the payoff. `cover(m, m, σ) == win(m, σ)` exactly, because a
pick'em spread *is* a zero margin — so the cover number and the win number can
never disagree about the same game.

`cover_prob` returns **None** with no real line. CFB's bug was inventing a 0.5
spread to cover against; a placeholder line is the same fabrication one level
up, so the honest answer is that there is no cover chance.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

#: For Normal errors, σ = MAE * sqrt(pi/2). Using MAE as if it were σ would
#: understate the spread by ~25% and make every probability too confident.
MAE_TO_SIGMA = float(np.sqrt(np.pi / 2))


def _checked_sigma(sigma: float, name: str = "sigma") -> float:
    """σ must be a positive, finite number. `x / 0` becomes ±inf and a NaN σ
    becomes a NaN probability that renders as a real number on screen."""
    value = float(sigma)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {sigma!r}")
    return value


def _cdf(x: float, sigma: float) -> float:
    return float(norm.cdf(x / _checked_sigma(sigma)))


def win_prob(predicted_margin: float, sigma: float) -> float:
    """P(home wins). Home winning *is* a positive margin, so this is the same
    CDF as cover at a pick'em line."""
    return _cdf(predicted_margin, sigma)


def cover_prob(predicted_margin: float, line: float | None, sigma: float) -> float | None:
    """P(home covers `line`). **None** when there is no real line — the caller
    must not render a cover chance it invented."""
    if line is None or not np.isfinite(float(line)):
        return None
    return _cdf(predicted_margin - float(line), sigma)


def over_prob(predicted_total: float, line: float | None, sigma: float) -> float | None:
    """P(total goes over `line`). None without a real line, as above."""
    if line is None or not np.isfinite(float(line)):
        return None
    return _cdf(predicted_total - float(line), sigma)


def fit_residual_sigma(errors) -> float:
    """σ from realised errors. Measured, never configured.

    Raises on an empty or zero-spread error set: that means the fit produced
    nothing usable, and returning 0.0 would make every probability exactly 0 or
    1.
    """
    values = np.asarray(list(errors), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite residuals to fit a sigma from")
    mae = float(np.mean(np.abs(values)))
    if mae <= 0.0:
        raise ValueError(f"residuals have zero spread (MAE {mae}); the fit is broken")
    return mae * MAE_TO_SIGMA


def residual_sigmas(*, margin_errors, total_errors) -> dict:
    """One σ per output, from pooled walk-forward residuals."""
    return {
        "margin_sigma": fit_residual_sigma(margin_errors),
        "total_sigma": fit_residual_sigma(total_errors),
    }