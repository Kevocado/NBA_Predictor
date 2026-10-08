"""Probability calibrators, and the rule for when they may be fitted.

The calibration gap on this model is a bias, not noise: in the middle of the
probability range it over-predicts home wins by about 7 points. That is what a
calibrator is for. What a calibrator is also for, if fitted carelessly, is
making a number look better than it is.

**The rule.** A calibrator is fitted inside the walk-forward. For window k it
sees out-of-fold predictions from windows *strictly before* k and nothing else;
it then maps window k's test predictions. No test game's own prediction is ever
part of the data that calibrates it.

**The first windows.** Window 0 has no earlier out-of-fold predictions to fit
on, and window 1 has only one window's worth. Both are left as
`IDENTITY_CALIBRATION` and the fact is recorded in the returned summary
(`calibrated_from_window`), so a pooled figure that mixes calibrated and
uncalibrated windows says which is which instead of quietly averaging them.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

#: How many earlier windows a calibrator needs before it is trusted. Below this
#: the fit is on too few games to be worth more than the bias it removes: at 250
#: games a two-parameter Platt fit carries roughly +/-0.03 of its own noise, and
#: isotonic is worse still because its effective degrees of freedom grow as the
#: steps get finer.
MIN_WINDOWS_FOR_CALIBRATION = 2


def IDENTITY_CALIBRATION(y: np.ndarray, p: np.ndarray) -> np.ndarray:  # noqa: N802
    """The calibrator that does nothing. Returned as a callable so every other
    calibrator and this one are interchangeable at the call site."""
    return np.asarray(p, dtype=float)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def _fit_platt(y: np.ndarray, p: np.ndarray):
    """Logistic regression on the logit of the prediction.

    Platt's method, and deliberately the two-parameter version. An intercept-only
    fit would only remove a constant offset; the slope is what also corrects a
    model that is too flat or too steep across the range.
    """
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(_logit(p).reshape(-1, 1), y)
    slope = float(lr.coef_[0][0])
    intercept = float(lr.intercept_[0])

    def apply(scores: np.ndarray) -> np.ndarray:
        return np.clip(_sigmoid(slope * _logit(scores) + intercept), 1e-12, 1 - 1e-12)

    return apply


def _fit_isotonic(y: np.ndarray, p: np.ndarray):
    """A step function from prediction to observed rate.

    More flexible than Platt and correspondingly more able to fit noise: with
    `out_of_bounds="clip"` it is flat outside the fitted range, so it cannot
    extrapolate a wild probability from a narrow one.
    """
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.asarray(p, dtype=float), y)

    def apply(scores: np.ndarray) -> np.ndarray:
        return np.clip(iso.predict(np.asarray(scores, dtype=float)), 1e-12, 1 - 1e-12)

    return apply


_FITTERS = {"platt": _fit_platt, "isotonic": _fit_isotonic, "identity": None}


def platt_calibrator():
    """Factory for the Platt calibrator, matching `isotonic_calibrator`'s shape."""
    return lambda y, p: fit_calibrator("platt", y, p)


def isotonic_calibrator():
    """Factory for the isotonic calibrator."""
    return lambda y, p: fit_calibrator("isotonic", y, p)


def fit_calibrator(kind: str, y: np.ndarray, p: np.ndarray):
    """Return a callable mapping scores to calibrated probabilities.

    `kind` is "platt", "isotonic" or "identity". A degenerate fit set -- one
    class of outcome -- raises rather than quietly returning the identity: a
    window that cannot be calibrated should be visible, and a silently
    uncalibrated window is the exact thing that makes a pooled number
    uninterpretable.
    """
    if kind not in _FITTERS:
        raise ValueError(f"unknown calibrator {kind!r}; expected one of {sorted(_FITTERS)}")

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"y and p must align: {y.shape} vs {p.shape}")
    if len(np.unique(y)) < 2:
        raise ValueError(
            "cannot calibrate on one class of outcome; a window with no home wins "
            "(or none away) has nothing to fit a monotone map to"
        )

    fitter = _FITTERS[kind]
    return IDENTITY_CALIBRATION if fitter is None else fitter(y, p)
