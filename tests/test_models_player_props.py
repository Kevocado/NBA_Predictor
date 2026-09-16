import numpy as np
import pandas as pd


def _synthetic_player_dataset(n=100, seed=1):
    rng = np.random.default_rng(seed)
    minutes = rng.uniform(15, 38, n)
    usage = rng.uniform(0.12, 0.32, n)
    X = pd.DataFrame({"minutes_roll": minutes, "usage_rate_roll": usage})
    y_points = minutes * usage * 40 + rng.normal(0, 3, n)
    y_double_double = (minutes > 30).astype(int)
    return X, pd.Series(y_points), pd.Series(y_double_double)


def test_stat_targets_lists_four_stats():
    from nba_predictor.models.player_props import STAT_TARGETS

    assert STAT_TARGETS == ["points", "rebounds", "assists", "threes"]


def test_train_player_stat_model_predicts_numeric_values():
    from nba_predictor.models.player_props import (
        predict_player_stat,
        train_player_stat_model,
    )

    X, y_points, _ = _synthetic_player_dataset()
    model = train_player_stat_model(X, y_points, n_estimators=20, max_depth=2)
    predictions = predict_player_stat(model, X)

    assert len(predictions) == len(X)
    assert np.isfinite(predictions).all()


def test_train_double_double_model_predicts_valid_probabilities():
    from nba_predictor.models.player_props import (
        predict_double_double_probability,
        train_double_double_model,
    )

    X, _, y_double_double = _synthetic_player_dataset()
    model = train_double_double_model(X, y_double_double, n_estimators=20, max_depth=2)
    probs = predict_double_double_probability(model, X)

    assert len(probs) == len(X)
    assert ((probs >= 0) & (probs <= 1)).all()
