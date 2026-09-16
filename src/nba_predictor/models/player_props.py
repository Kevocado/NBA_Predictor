import numpy as np
import pandas as pd
import xgboost as xgb

STAT_TARGETS = ["points", "rebounds", "assists", "threes"]


def train_player_stat_model(X: pd.DataFrame, y: pd.Series, **xgb_params) -> xgb.XGBRegressor:
    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, **xgb_params}
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def train_double_double_model(X: pd.DataFrame, y: pd.Series, **xgb_params) -> xgb.XGBClassifier:
    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, **xgb_params}
    model = xgb.XGBClassifier(**params, eval_metric="logloss")
    model.fit(X, y)
    return model


def predict_player_stat(model: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


def predict_double_double_probability(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
