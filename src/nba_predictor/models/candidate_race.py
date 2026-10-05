import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier, Ridge

from nba_predictor.models.evaluate.walk_forward_eval import (
    walk_forward_metrics,
    walk_forward_regression,
)
from nba_predictor.features.build import FEATURE_COLUMNS


FEATURE_COLS = ["strength"]
WIN_MODEL_CANDIDATES = ["ridge", "xgboost", "elo"]


def _set_seeds():
    np.random.seed(42)
    try:
        import xgboost as xgb

        xgb.set_config(verbosity=0)
    except Exception:
        pass


def make_ridge_classifier_factory(feature_cols=None):
    cols = feature_cols or FEATURE_COLS

    def factory(train_df):
        from sklearn.linear_model import RidgeClassifier
        model = RidgeClassifier(random_state=42)
        model.fit(train_df[cols], train_df["home_win"])
        try:
            return lambda X: model.predict_proba(X[cols])[:, 1]
        except Exception:
            try:
                return lambda X: model._predict_proba_lr(X[cols])[:, 1]
            except Exception:
                def _pred(X):
                    d = model.decision_function(X[cols])
                    from scipy.special import expit
                    p = expit(d)
                    return np.clip(p, 1e-12, 1 - 1e-12)
                return _pred

    return factory


def make_ridge_regressor_factory(target="home_margin", feature_cols=None):
    cols = feature_cols or FEATURE_COLS

    def factory(train_df):
        model = Ridge(random_state=42)
        y = train_df[target]
        model.fit(train_df[cols], y)
        return lambda X: model.predict(X[cols])

    return factory


def make_xgb_classifier_factory(feature_cols=None):
    import xgboost as xgb

    cols = feature_cols or FEATURE_COLS

    def factory(train_df):
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(train_df[cols], train_df["home_win"])
        return lambda X: model.predict_proba(X[cols])[:, 1]

    return factory


def make_xgb_regressor_factory(target="home_margin", feature_cols=None):
    import xgboost as xgb

    cols = feature_cols or FEATURE_COLS

    def factory(train_df):
        model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
        if target in train_df:
            y = train_df[target]
        elif target == "home_margin" and {"home_pts", "away_pts"}.issubset(train_df.columns):
            y = train_df["home_pts"] - train_df["away_pts"]
        elif target == "home_total" and {"home_pts", "away_pts"}.issubset(train_df.columns):
            y = train_df["home_pts"] + train_df["away_pts"]
        else:
            y = train_df[cols[0]] if cols else 0
        model.fit(train_df[cols], y)
        return lambda X: model.predict(X[cols])

    return factory


def make_elo_factory(feature_cols=None):
    cols = feature_cols or FEATURE_COLS

    def factory(train_df):
        train_df = train_df.sort_values("game_date").reset_index(drop=True)
        home_ratings = {}
        away_ratings = {}
        k = 20.0
        home_adv = 100.0

        all_teams = set(train_df["home_team"].tolist() + train_df["away_team"].tolist())
        for t in all_teams:
            home_ratings[t] = 1500.0
            away_ratings[t] = 1500.0

        for _, row in train_df.iterrows():
            h = row["home_team"]
            a = row["away_team"]
            rh = home_ratings.get(h, 1500.0) + home_adv
            ra = away_ratings.get(a, 1500.0)
            eh = 1.0 / (1.0 + 10.0 ** ((ra - rh) / 400.0))
            res_h = float(row["home_win"])
            home_ratings[h] = rh - home_adv + k * (res_h - eh)
            away_ratings[a] = ra + k * ((1.0 - res_h) - (1.0 - eh))

        def predict_fn(X):
            preds = []
            for _, row in X.iterrows():
                h = row["home_team"]
                a = row["away_team"]
                rh = home_ratings.get(h, 1500.0) + home_adv
                ra = away_ratings.get(a, 1500.0)
                eh = 1.0 / (1.0 + 10.0 ** ((ra - rh) / 400.0))
                preds.append(float(np.clip(eh, 1e-12, 1 - 1e-12)))
            return np.array(preds)

        return predict_fn

    return factory


def run_candidate_race(df, candidates=None, feature_cols=None, windows=4, date_col="game_date"):
    _set_seeds()
    df = df.copy().sort_values(date_col).reset_index(drop=True)

    cols = feature_cols or (["strength"] if "strength" in df.columns else FEATURE_COLUMNS)

    if candidates is None:
        candidates = {
            "ridge": make_ridge_classifier_factory(cols),
            "xgboost": make_xgb_classifier_factory(cols),
            "elo": make_elo_factory(cols),
        }

    results = {}
    win_factories = {}
    margin_factories = {}
    total_factories = {}

    for name, factory in candidates.items():
        key = name.lower()
        if key == "xgb":
            key = "xgboost"
        results[key] = {}
        win_factories[key] = factory
        if key == "ridge":
            margin_factories[key] = make_ridge_regressor_factory("home_margin", cols)
            total_factories[key] = make_ridge_regressor_factory("home_total", cols)
        elif key == "xgboost":
            margin_factories[key] = make_xgb_regressor_factory("home_margin", cols)
            total_factories[key] = make_xgb_regressor_factory("home_total", cols)
        else:
            margin_factories[key] = make_ridge_regressor_factory("home_margin", cols)
            total_factories[key] = make_ridge_regressor_factory("home_total", cols)

    for key, factory in win_factories.items():
        try:
            wf_win = walk_forward_metrics(df, model_factory=factory, windows=windows, date_col=date_col)
            results[key]["win"] = wf_win
        except Exception as e:
            results[key]["win"] = {"pooled": {"log_loss": float("inf")}, "error": str(e)}

    for key, factory in margin_factories.items():
        try:
            if {"home_pts", "away_pts"}.issubset(df.columns):
                df_margin = df.assign(home_margin=df["home_pts"] - df["away_pts"])
            else:
                df_margin = df.copy()
            if "home_margin" not in df_margin.columns and "strength" in df_margin.columns:
                df_margin = df_margin.assign(home_margin=df_margin["strength"])
            wf_margin = walk_forward_regression(df_margin, model_factory=factory, target="home_margin", windows=windows, date_col=date_col)
            results[key]["margin"] = wf_margin
        except Exception:
            results[key]["margin"] = {"pooled": {"mae": float("inf")}}

    for key, factory in total_factories.items():
        try:
            if {"home_pts", "away_pts"}.issubset(df.columns):
                df_total = df.assign(home_total=df["home_pts"] + df["away_pts"])
            else:
                df_total = df.copy()
            if "home_total" not in df_total.columns and "strength" in df_total.columns:
                df_total = df_total.assign(home_total=df_total["strength"] + 200.0)
            wf_total = walk_forward_regression(df_total, model_factory=factory, target="home_total", windows=windows, date_col=date_col)
            results[key]["total"] = wf_total
        except Exception:
            results[key]["total"] = {"pooled": {"mae": float("inf")}}

    servable_keys = []
    for k in results:
        kl = k.lower()
        if kl in ("ridge", "xgboost"):
            servable_keys.append(kl)
    if not servable_keys:
        servable_keys = ["ridge", "xgboost"]

    best = None
    best_loss = float("inf")
    for k in servable_keys:
        loss = results.get(k, {}).get("win", {}).get("pooled", {}).get("log_loss", float("inf"))
        if best is None or loss < best_loss - 1e-18:
            best_loss = loss
            best = k
    winner = best or servable_keys[0]

    return {"winner": winner, "results": results}
