import numpy as np
import pandas as pd


def make_player_games(n=200, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-10-01", periods=n, freq="D")
    rows = []
    for i in range(n):
        rows.append(
            {
                "player_id": f"p{i % 20}",
                "game_id": f"g{i}",
                "game_date": dates[i],
                "team": "A",
                "opponent": "B",
                "minutes": rng.uniform(20, 35),
                "points": rng.uniform(10, 30),
                "rebounds": rng.uniform(3, 12),
                "assists": rng.uniform(1, 8),
                "fg3m": rng.uniform(0, 4),
                "threes": rng.uniform(0, 4),
            }
        )
    return pd.DataFrame(rows)


def dummy_prop_factories():
    from nba_predictor.models.player_props import train_player_stat_model

    def factory_points(X, y):
        return train_player_stat_model(X, y, n_estimators=10, max_depth=2)

    def factory_rebounds(X, y):
        return train_player_stat_model(X, y, n_estimators=10, max_depth=2)

    def factory_assists(X, y):
        return train_player_stat_model(X, y, n_estimators=10, max_depth=2)

    def factory_threes(X, y):
        return train_player_stat_model(X, y, n_estimators=10, max_depth=2)

    return {
        "points": factory_points,
        "rebounds": factory_rebounds,
        "assists": factory_assists,
        "threes": factory_threes,
    }

from nba_predictor.models.prop_holdout import prop_holdout_metrics


def test_holdout_is_chronological_and_reports_both():
    df = make_player_games(n=200)
    res = prop_holdout_metrics(df, factories=dummy_prop_factories())
    for market in ("points", "rebounds", "assists", "threes"):
        assert res[market]["holdout_mae"] >= 0
        assert "in_sample_mae" in res[market]
        # holdout window strictly after train window
        assert res[market]["train_max_date"] < res[market]["holdout_min_date"]
        assert res[market]["holdout_mae"] > 0
        assert res[market]["in_sample_mae"] >= 0
        assert isinstance(res[market]["n_train"], int)
        assert isinstance(res[market]["n_holdout"], int)
    # all four markets reported
    assert set(res.keys()) == {"points", "rebounds", "assists", "threes"}
