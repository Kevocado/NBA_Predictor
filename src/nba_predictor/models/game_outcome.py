import numpy as np
import pandas as pd
import xgboost as xgb


def train_win_probability_model(X: pd.DataFrame, y: pd.Series, **xgb_params) -> xgb.XGBClassifier:
    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, **xgb_params}
    model = xgb.XGBClassifier(**params, eval_metric="logloss")
    model.fit(X, y)
    return model


def train_win_probability_model_ridge(X: pd.DataFrame, y: pd.Series, **kwargs):
    from sklearn.linear_model import RidgeClassifier

    model = RidgeClassifier(random_state=42)
    model.fit(X, y)
    return model


def train_margin_model(X: pd.DataFrame, y: pd.Series, **xgb_params) -> xgb.XGBRegressor:
    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, **xgb_params}
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def train_margin_model_ridge(X: pd.DataFrame, y: pd.Series, **kwargs):
    from sklearn.linear_model import Ridge

    model = Ridge(random_state=42)
    model.fit(X, y)
    return model


def train_total_model(X: pd.DataFrame, y: pd.Series, **xgb_params) -> xgb.XGBRegressor:
    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, **xgb_params}
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def train_total_model_ridge(X: pd.DataFrame, y: pd.Series, **kwargs):
    from sklearn.linear_model import Ridge

    model = Ridge(random_state=42)
    model.fit(X, y)
    return model


def predict_win_probability(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        try:
            return model._predict_proba_lr(X)[:, 1]
        except Exception:
            d = model.decision_function(X)
            from scipy.special import expit
            return np.clip(expit(d), 1e-12, 1 - 1e-12)
