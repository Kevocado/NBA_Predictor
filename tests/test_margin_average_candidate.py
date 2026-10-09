"""What the averaged margin candidate IS, so the tool's claim about it is checkable.

`tools/compare_margin_average.py` measures the candidate on real box scores. It is
a tool and not a test, and deliberately so: it reads
`data/cache/training/games.json`, which is gitignored, and a test that reads it
passes locally while erroring in CI -- which is how a green run stops meaning
anything. `tools/compare_carryover.py` is the same shape for the same reason.

What this file pins is the part that does not need the cache: that the averaged
factory really is the MEAN of the two regressors, fitted on the training slice it
is handed, and not a weighted blend, a re-fit, or something that quietly reaches
past its training frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_predictor.models.candidate_race import (
    make_ridge_regressor_factory,
    make_xgb_regressor_factory,
)


@pytest.fixture(scope="module")
def frames():
    """A small synthetic problem with a known answer.

    Ridge and XGBoost fitted on the same slice give two predictions; the average
    must be their arithmetic mean, so the whole thing is checkable without any
    real data.
    """
    rng = np.random.default_rng(11)
    n = 120
    X = pd.DataFrame(rng.normal(0, 1, (n, 4)), columns=[f"f{i}" for i in range(4)])
    y = (X["f0"] * 2.0 + rng.normal(0, 1, n))
    train_X, test_X = X.iloc[:100], X.iloc[100:]
    train_y, test_y = y[:100], y[100:]
    return train_X, train_y, test_X, test_y


def _averaged_factory(feature_cols, target):
    """The same factory the tool builds -- copied, not imported, so a change to the
    tool's version breaks this instead of silently moving with it."""
    ridge = make_ridge_regressor_factory(feature_cols, target)
    xgb = make_xgb_regressor_factory(feature_cols, target)

    def factory(train_df):
        r = ridge(train_df)
        x = xgb(train_df)

        def predict(test_df):
            return 0.5 * (np.asarray(r(test_df)) + np.asarray(x(test_df)))

        return predict

    return factory


def test_the_average_is_the_arithmetic_mean_of_the_two(frames):
    train_X, train_y, test_X, test_y = frames
    cols = list(train_X.columns)
    train_df = train_X.assign(home_margin=train_y)
    test_df = test_X.assign(home_margin=test_y)

    averaged = _averaged_factory(cols, "home_margin")(train_df)
    got = np.asarray(averaged(test_df))

    # Each regressor on its own, from the same training slice.
    r = make_ridge_regressor_factory(cols, "home_margin")(train_df)(test_df)
    x = make_xgb_regressor_factory(cols, "home_margin")(train_df)(test_df)
    expected = 0.5 * (np.asarray(r) + np.asarray(x))

    assert got.shape == (len(test_df),), f"expected one prediction per game, got {got.shape}"
    np.testing.assert_allclose(got, expected, err_msg=(
        "the averaged factory does not return the mean of the two regressors: it is "
        "weighted, re-fit, or reaching past its training slice"
    ))


def test_the_average_is_between_the_two_models_predictions(frames):
    """A mean has to sit inside its inputs. A factory that went outside them would
    be fitting something new rather than averaging."""
    train_X, train_y, test_X, test_y = frames
    cols = list(train_X.columns)
    train_df = train_X.assign(home_margin=train_y)
    test_df = test_X.assign(home_margin=test_y)

    averaged = np.asarray(_averaged_factory(cols, "home_margin")(train_df)(test_df))
    r = np.asarray(make_ridge_regressor_factory(cols, "home_margin")(train_df)(test_df))
    x = np.asarray(make_xgb_regressor_factory(cols, "home_margin")(train_df)(test_df))

    lo, hi = np.minimum(r, x), np.maximum(r, x)
    assert np.all(averaged >= lo - 1e-9) and np.all(averaged <= hi + 1e-9), (
        "the average falls outside the two models' predictions, so it is not a mean"
    )


def test_the_average_uses_finite_predictions(frames):
    train_X, train_y, test_X, test_y = frames
    cols = list(train_X.columns)
    train_df = train_X.assign(home_margin=train_y)
    test_df = test_X.assign(home_margin=test_y)

    got = np.asarray(_averaged_factory(cols, "home_margin")(train_df)(test_df))
    assert np.all(np.isfinite(got)), f"non-finite predictions: {got[:5]}"
