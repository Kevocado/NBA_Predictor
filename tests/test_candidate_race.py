import numpy as np
import pandas as pd


def make_games(n: int = 120) -> pd.DataFrame:
    """n games on consecutive dates with a home-win label the factory can learn."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2025-10-22", periods=n).astype(str)
    strength = np.linspace(-2.0, 2.0, n) + rng.normal(0, 0.2, n)
    p_home = 1 / (1 + np.exp(-strength))
    teams = ["BOS", "MIA", "LAL", "GSW"]
    return pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(n)],
            "game_date": dates,
            "home_team": [teams[i % 4] for i in range(n)],
            "away_team": [teams[(i + 1) % 4] for i in range(n)],
            "strength": strength,
            "home_win": (rng.random(n) < p_home).astype(int),
        }
    )


def ridge_factory(train_df: pd.DataFrame):
    from sklearn.linear_model import RidgeClassifier

    FEATURE_COLS = ["strength"]

    def predict_fn(X):
        model = RidgeClassifier(random_state=42)
        model.fit(train_df[FEATURE_COLS], train_df["home_win"])
        return model.predict_proba(X[FEATURE_COLS])[:, 1]

    return predict_fn


def xgb_factory(train_df: pd.DataFrame):
    import xgboost as xgb

    FEATURE_COLS = ["strength"]
    model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")
    model.fit(train_df[FEATURE_COLS], train_df["home_win"])
    return lambda X: model.predict_proba(X[FEATURE_COLS])[:, 1]


def test_race_picks_lower_walk_forward_logloss():
    from nba_predictor.models.candidate_race import run_candidate_race

    df = make_games(n=120)
    res = run_candidate_race(df)
    assert res["winner"] in ("ridge", "xgboost", "xgb")
    results = res["results"]
    servable = {k: v for k, v in results.items() if k in ("ridge", "xgboost", "xgb")}
    losses = {k: v["win"]["pooled"]["log_loss"] for k, v in servable.items()}
    norm = {}
    for k, l in losses.items():
        kk = "xgboost" if k == "xgb" else k
        norm[kk] = l
    winner = res["winner"]
    if winner == "xgb":
        winner = "xgboost"
    assert norm[winner] == min(norm.values())


def test_factory_receives_train_sizes_increasing():
    from nba_predictor.models.candidate_race import run_candidate_race
    from nba_predictor.models.evaluate import walk_forward_eval

    received = []
    def record_factory(train_df):
        received.append(len(train_df))
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(train_df[["strength"]], train_df["home_win"])
        return lambda X: model.predict_proba(X[["strength"]])[:, 1]

    df = make_games(120)
    walk_forward_eval.walk_forward_metrics(df, model_factory=record_factory, windows=4)
    assert len(received) == 4
    assert received == sorted(received)
    assert received[0] < received[-1]


def test_race_deterministic():
    from nba_predictor.models.candidate_race import run_candidate_race

    df = make_games(120)
    res1 = run_candidate_race(df)
    res2 = run_candidate_race(df)
    assert res1 == res2


def test_race_includes_elo_as_reference_only():
    from nba_predictor.models.candidate_race import run_candidate_race

    df = make_games(n=120)
    res = run_candidate_race(df)
    assert "elo" in res["results"]
    assert res["winner"] != "elo"
