import numpy as np
import pandas as pd
import pytest


def _synthetic_dataset(n=120, seed=0):
    rng = np.random.default_rng(seed)
    power_diff = rng.normal(0, 100, n)
    X = pd.DataFrame(
        {
            "power_rating_diff": power_diff,
            "home_rest_days": rng.integers(0, 4, n),
            "away_rest_days": rng.integers(0, 4, n),
        }
    )
    win_prob = 1 / (1 + np.exp(-power_diff / 100))
    y_win = (rng.random(n) < win_prob).astype(int)
    y_margin = power_diff / 20 + rng.normal(0, 5, n)
    y_total = 220 + rng.normal(0, 10, n)
    return X, pd.Series(y_win), pd.Series(y_margin), pd.Series(y_total)


def test_train_win_probability_model_predicts_valid_probabilities():
    from nba_predictor.models.game_outcome import (
        predict_win_probability,
        train_win_probability_model,
    )

    X, y_win, _, _ = _synthetic_dataset()
    model = train_win_probability_model(X, y_win, n_estimators=20, max_depth=2)
    probs = predict_win_probability(model, X)

    assert len(probs) == len(X)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_train_margin_model_returns_numeric_predictions():
    from nba_predictor.models.game_outcome import train_margin_model

    X, _, y_margin, _ = _synthetic_dataset()
    model = train_margin_model(X, y_margin, n_estimators=20, max_depth=2)
    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert np.isfinite(predictions).all()


def test_train_total_model_returns_numeric_predictions():
    from nba_predictor.models.game_outcome import train_total_model

    X, _, _, y_total = _synthetic_dataset()
    model = train_total_model(X, y_total, n_estimators=20, max_depth=2)
    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert np.isfinite(predictions).all()
