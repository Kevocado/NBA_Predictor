"""The candidate race: pick the winner, then serve it (spec section 3.1, section 6).

The incumbent was three fixed XGBoost models (`game_outcome.py`) chosen by
assumption. On the real 1,390-game set the win model scores walk-forward
log-loss 0.7059 against a naive 0.6868 -- worse than doing nothing clever. So
the candidates race on expanding-window walk-forward and the manifest records
who won. NFL's precedent is that the linear model beat the booster at this
sample size; that is a hypothesis to test here, not an assumption to build in.

Three things this module refuses to do, each because a rejected attempt did it:

* **No silent scoring of a broken candidate.** If a candidate raises, the race
  raises. Catching it and writing `inf` makes the survivor the winner by
  default, and the "winner" becomes an artifact of the breakage.
* **No constant masquerading as a model.** Elo is a real rating loop.
* **No closure over an already-fitted model.** Every factory fits on the window
  it is handed; that is the leakage Task 4 shipped.

The caller supplies the frame. Re-deriving rolling features here would score a
different dataset than the one the served model was fitted on.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from nba_predictor.features.build import FEATURE_COLUMNS
from nba_predictor.models.evaluate.walk_forward_eval import (
    expanding_windows,
    walk_forward_metrics,
    walk_forward_regression,
)

#: Elo's conventional rating scale. The K/400 form is standard Elo; these are
#: the values the rating maths is written against, not tuned parameters.
ELO_SCALE = 400.0
ELO_K = 20.0

RANDOM_STATE = 42


def _seed() -> None:
    np.random.seed(RANDOM_STATE)


# --------------------------------------------------------------------------
# Factories. Each is `f(feature_cols, ...) -> factory`, where `factory(train_df)`
# fits on `train_df` and returns a predictor taking the full frame.
# --------------------------------------------------------------------------


def make_logistic_classifier_factory(feature_cols: list[str]) -> Callable:
    """Linear win model. `LogisticRegression` rather than `RidgeClassifier`
    because the latter has no `predict_proba` and cannot be scored on
    log-loss at all -- a race it cannot enter is not a win.

    Handles single-class training data by returning a constant predictor.
    """

    def factory(train_df: pd.DataFrame):
        from sklearn.linear_model import LogisticRegression

        _seed()
        y = train_df["home_win"]
        classes = y.unique()
        if len(classes) == 1:
            # Single class: return constant probability equal to the class prior
            const_prob = float(classes[0])
            return lambda X: np.full(len(X), const_prob)

        _seed()
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(train_df[feature_cols], y)
        return lambda X: model.predict_proba(X[feature_cols])[:, 1]

    return factory


def make_ridge_regressor_factory(feature_cols: list[str], target_col: str) -> Callable:
    def factory(train_df: pd.DataFrame):
        from sklearn.linear_model import Ridge

        _seed()
        model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        model.fit(train_df[feature_cols], train_df[target_col])
        return lambda X: model.predict(X[feature_cols])

    return factory


def make_xgb_classifier_factory(feature_cols: list[str]) -> Callable:
    def factory(train_df: pd.DataFrame):
        import xgboost as xgb

        _seed()
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        )
        model.fit(train_df[feature_cols], train_df["home_win"])
        return lambda X: model.predict_proba(X[feature_cols])[:, 1]

    return factory


def make_xgb_regressor_factory(feature_cols: list[str], target_col: str) -> Callable:
    def factory(train_df: pd.DataFrame):
        import xgboost as xgb

        _seed()
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        )
        model.fit(train_df[feature_cols], train_df[target_col])
        return lambda X: model.predict(X[feature_cols])

    return factory


def elo_probability_factory(k: float = ELO_K, scale: float = ELO_SCALE) -> Callable:
    """Win-only reference: a real rating loop, refit per window like the rest.

    Ratings start at 1500 and update game by game in date order,
    `R' = R + k * (result - expected)`. The returned predictor scores test rows
    from the ratings the training window left behind, which is all a
    pre-game Elo book can know.

    A constant 0.5 is not Elo. It would also have *won* this race -- 0.6931
    beats XGBoost's 0.7059 -- which is how a coin flip gets reported as a model.
    """
    initial = scale * 0.5  # 200 at scale 400, i.e. the conventional 1500

    def factory(train_df: pd.DataFrame):
        ratings: dict[str, float] = {}

        def rating(team: str) -> float:
            return ratings.setdefault(team, initial)

        ordered = train_df.sort_values("game_date")
        for row in ordered.itertuples():
            home, away = row.home_team, row.away_team
            expected = 1.0 / (1.0 + scale ** (-(rating(home) - rating(away)) / scale))
            delta = k * (row.home_win - expected)
            ratings[home] = rating(home) + delta
            ratings[away] = rating(away) - delta

        def predict(X: pd.DataFrame) -> np.ndarray:
            return np.array(
                [
                    1.0
                    / (
                        1.0
                        + scale ** (-(rating(h) - rating(a)) / scale)
                    )
                    for h, a in zip(X["home_team"], X["away_team"])
                ],
                dtype=float,
            )

        return predict

    return factory


#: Reference-only: scored, reported, never served (spec section 3.1).
REFERENCE_ONLY = ("elo",)


def default_candidates(feature_cols: list[str]) -> dict:
    return {
        "logistic": {"win": make_logistic_classifier_factory(feature_cols)},
        "ridge": {
            "margin": make_ridge_regressor_factory(feature_cols, "home_margin"),
            "total": make_ridge_regressor_factory(feature_cols, "home_total"),
        },
        "xgboost": {
            "win": make_xgb_classifier_factory(feature_cols),
            "margin": make_xgb_regressor_factory(feature_cols, "home_margin"),
            "total": make_xgb_regressor_factory(feature_cols, "home_total"),
        },
        "elo": {"win": elo_probability_factory()},
    }


#: The metric that decides each target's winner. Walk-forward pooled, per target.
DECIDING_METRIC = {"win": "log_loss", "margin": "mae", "total": "mae"}


def _pick_winner(scored: dict, target: str) -> str | None:
    metric = DECIDING_METRIC[target]
    contenders = {
        name: res[target]["pooled"][metric]
        for name, res in scored.items()
        if target in res and name not in REFERENCE_ONLY
    }
    if not contenders:
        return None
    return min(contenders, key=contenders.get)


def run_candidate_race(
    df: pd.DataFrame,
    candidates: dict | None = None,
    feature_cols: list[str] | None = None,
    windows: int = 4,
    date_col: str = "game_date",
) -> dict:
    """Race every candidate on expanding-window walk-forward. Winner per target.

    No candidate is wrapped in a try/except. A candidate that cannot be scored
    raises, because a race that silently drops its slowest entrant reports a
    winner chosen by the breakage rather than by the evidence.

    Naive baselines ride along inside each pooled result (`coinflip_log_loss`,
    `naive_log_loss`, `naive_mae`, `naive_mae_fixed`) -- all three targets, so
    no comparison in the report is missing its baseline.
    """
    _seed()
    df = df.sort_values(date_col).reset_index(drop=True)

    for required in ("home_pts", "away_pts", "home_win"):
        if required not in df.columns:
            raise ValueError(f"{required} missing; pass the training frame, not raw games")
    df = df.assign(
        home_margin=df["home_pts"] - df["away_pts"],
        home_total=df["home_pts"] + df["away_pts"],
    )

    feature_cols = list(feature_cols or FEATURE_COLUMNS)
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise ValueError("no feature columns present in the frame")

    candidates = candidates or default_candidates(feature_cols)

    # Two distinct naive comparators per regression target: the per-window
    # training mean (adapts), and a fixed constant (does not). The total's fixed
    # constant is taken from the FIRST training window only and then held, so it
    # is a league average known early rather than a mean refitted per window --
    # which would make it the same number as `naive_mae` wearing another name.
    first_train_idx = expanding_windows(df[date_col], windows)[0][0]
    fixed_baselines = {
        "home_margin": 3.0,
        "home_total": float(df.iloc[first_train_idx]["home_total"].mean()),
    }

    results: dict[str, dict] = {}
    for name, cand in candidates.items():
        scored: dict[str, dict] = {}
        if "win" in cand:
            scored["win"] = walk_forward_metrics(
                df, model_factory=cand["win"], windows=windows,
                date_col=date_col, target_col="home_win",
            )
        for target in ("margin", "total"):
            if target in cand:
                scored[target] = walk_forward_regression(
                    df, model_factory=cand[target], target=f"home_{target}",
                    windows=windows, date_col=date_col,
                    fixed_baseline=fixed_baselines[f"home_{target}"],
                )
        results[name] = scored

    return {
        "winner": {t: _pick_winner(results, t) for t in DECIDING_METRIC},
        "results": results,
        "n_games": len(df),
        "feature_cols": feature_cols,
    }