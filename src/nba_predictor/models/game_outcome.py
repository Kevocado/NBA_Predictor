"""Fitted-model factories for the game outcome models.

These three models were fixed XGBoost by assumption (spec gap G2). The
candidate race on real data says otherwise: over expanding-window walk-forward
on 1,368 games, the linear model wins the win model (log-loss 0.6637 vs
XGBoost's 0.7059) and beats both naive baselines, and ridge wins margin. So
each factory takes the race winner as a `candidate`.

`candidate` defaults to the incumbent, so every existing caller keeps working
until the winner is threaded through explicitly. The XGBoost path stays
regardless -- a future race on more data may reverse this, and a deleted
incumbent leaves nothing to race back to.

`predict_win_probability` and the regressors' `.predict` are estimator-agnostic
on purpose: they accept anything exposing `predict_proba` / `predict`, which is
what lets the winner be swapped without rewriting serving.
"""

import numpy as np
import pandas as pd

#: Hyperparameters the incumbent was always fitted with (spec section 6).
XGB_PARAMS = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}


def _check_candidate(candidate: str) -> str:
    if candidate not in ("xgboost", "logistic", "ridge"):
        # A typo must not fall through to the incumbent: that is how a race
        # result gets quietly ignored.
        raise ValueError(
            f"unknown candidate {candidate!r}; expected 'xgboost', 'logistic' or 'ridge'"
        )
    return candidate


def train_win_probability_model(X: pd.DataFrame, y: pd.Series, candidate: str = "xgboost", **xgb_params):
    """Fit the win-probability model. `candidate` is the race winner's name."""
    _check_candidate(candidate)
    if candidate == "logistic":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        import xgboost as xgb

        model = xgb.XGBClassifier(
            **{**XGB_PARAMS, **xgb_params},
            random_state=42,
            eval_metric="logloss",
        )
    model.fit(X, y)
    return model


def train_margin_model(X: pd.DataFrame, y: pd.Series, candidate: str = "xgboost", **xgb_params):
    _check_candidate(candidate)
    if candidate == "ridge":
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0, random_state=42)
    else:
        import xgboost as xgb

        model = xgb.XGBRegressor(**{**XGB_PARAMS, **xgb_params}, random_state=42)
    model.fit(X, y)
    return model


def train_total_model(X: pd.DataFrame, y: pd.Series, candidate: str = "xgboost", **xgb_params):
    return train_margin_model(X, y, candidate=candidate, **xgb_params)


def predict_win_probability(model, X: pd.DataFrame) -> np.ndarray:
    """Estimator-agnostic on purpose -- see the module docstring."""
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)