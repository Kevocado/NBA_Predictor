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
    # No callers. Deliberately not served: an unserved, uncalibrated classifier
    # is not a probability this project may show. Left alone on purpose.
    return model.predict_proba(X)[:, 1]


def in_sample_mae_by_stat(resolved_rows) -> dict[str, float | None]:
    """Mean absolute error per stat over RESOLVED rows only.

    A row is resolved when it carries both a prediction and an actual value.
    A prediction with no actual is not evidence about the model's accuracy --
    scoring it against nothing would drag the error toward the model's own
    guesses -- so it never contributes.

    A stat with no resolved rows is None, never 0.0. 0.0 would claim the model
    never missed by a tenth of a point, which is a different and much stronger
    statement than "never measured"; TrackRecordOut.hit_rate already reasons
    this way. The caller renders the difference; this module only refuses to
    make it up.

    In-sample, like the training metrics in pipeline/ingest.py: these rows are
    the same rows the models were fitted on. Stated rather than hidden, and the
    honest way to read a "+/-" beside an NBA projection.
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    for row in resolved_rows or ():
        stat = row["stat"]
        predicted = row["predicted_value"]
        actual = row["actual_value"]
        if predicted is None or actual is None:
            continue
        totals[stat] = totals.get(stat, 0.0) + abs(float(predicted) - float(actual))
        counts[stat] = counts.get(stat, 0) + 1

    return {
        stat: (totals[stat] / counts[stat] if counts.get(stat) else None)
        for stat in STAT_TARGETS
    }
