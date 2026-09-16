import numpy as np
import pandas as pd
import xgboost as xgb


def train_win_probability_model(X: pd.DataFrame, y: pd.Series, **xgb_params) -> xgb.XGBClassifier:
    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, **xgb_params}
    model = xgb.XGBClassifier(**params, eval_metric="logloss")
    model.fit(X, y)
    return model


def train_margin_model(X: pd.DataFrame, y: pd.Series, **xgb_params) -> xgb.XGBRegressor:
    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, **xgb_params}
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def train_total_model(X: pd.DataFrame, y: pd.Series, **xgb_params) -> xgb.XGBRegressor:
    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, **xgb_params}
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def predict_win_probability(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
