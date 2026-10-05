"""The serving path must follow the race, not the assumption.

The race on real data is not close: logistic beats XGBoost on the win model
(0.6637 vs 0.7059 walk-forward log-loss) and beats both naive baselines, while
XGBoost loses to both. Ridge likewise wins margin. Three fixed XGBoost models
were chosen by assumption in `game_outcome.py` (spec gap G2), and the race says
that assumption was wrong.

So the fitted-model factories take the race winner. `predict_win_probability`
already accepts anything exposing `predict_proba`, and the regressors anything
exposing `predict` -- which is why this is a small change rather than a
serving rewrite.

The loser's code stays. A future race on more data may reverse this, and
deleting the incumbent would leave nothing to race back to.
"""

import numpy as np
import pandas as pd
import pytest

from nba_predictor.models.game_outcome import (
    predict_win_probability,
    train_margin_model,
    train_total_model,
    train_win_probability_model,
)


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame(rng.normal(0, 1, (n, 4)), columns=list("abcd"))
    margin = 3 + X["a"] * 5 + rng.normal(0, 2, n)
    total = 218 + X["b"] * 4 + rng.normal(0, 5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-X["a"]))).astype(int)
    return X, y, margin, total


def test_win_factory_can_fit_the_race_winner(frames):
    """`candidate="logistic"` is what the race selected for the win model."""
    X, y, _, _ = frames
    model = train_win_probability_model(X, y, candidate="logistic")

    probs = predict_win_probability(model, X)
    assert probs.shape == (len(X),)
    assert ((probs >= 0) & (probs <= 1)).all()
    # A logistic on a real signal must rank it, not sit at the base rate.
    assert probs.std() > 0.05, f"probabilities are flat: std {probs.std():.4f}"


def test_regressor_factories_can_fit_the_race_winner(frames):
    X, _, margin, total = frames
    for factory, target in ((train_margin_model, margin), (train_total_model, total)):
        model = factory(X, target, candidate="ridge")
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()


def test_the_incumbent_xgboost_path_still_exists(frames):
    """Keep the loser. A future race on more data may reverse this result, and a
    deleted incumbent leaves nothing to race back to."""
    X, y, margin, _ = frames
    model = train_win_probability_model(X, y, candidate="xgboost")
    assert predict_win_probability(model, X).shape == (len(X),)
    assert train_margin_model(X, margin, candidate="xgboost").predict(X).shape == (len(X),)


def test_unknown_candidate_is_refused_loudly(frames):
    """A typo must not silently fall back to the incumbent -- that is how a
    race result gets quietly ignored."""
    X, y, _, _ = frames
    with pytest.raises(ValueError, match="candidate"):
        train_win_probability_model(X, y, candidate="lgistic")


def test_default_candidate_is_unchanged(frames):
    """The default stays the incumbent so every existing caller keeps working
    until the winner is threaded through explicitly."""
    X, y, _, _ = frames
    default = train_win_probability_model(X, y)
    explicit = train_win_probability_model(X, y, candidate="xgboost")
    np.testing.assert_allclose(
        predict_win_probability(default, X), predict_win_probability(explicit, X)
    )


def test_predict_accepts_either_model_without_knowing_which(frames):
    """`predict_win_probability` must not be typed to one estimator -- that is
    what lets the race winner be swapped in without touching serving."""
    X, y, _, _ = frames
    for candidate in ("xgboost", "logistic"):
        model = train_win_probability_model(X, y, candidate=candidate)
        assert np.isfinite(predict_win_probability(model, X)).all()