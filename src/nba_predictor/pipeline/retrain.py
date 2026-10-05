from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_predictor.features.build import build_training_frame
from nba_predictor.models.evaluate.walk_forward import chronological_split
from nba_predictor.models.evaluate.walk_forward_eval import (
    walk_forward_metrics,
    walk_forward_regression,
)
from nba_predictor.models.game_outcome import (
    predict_win_probability,
    train_margin_model,
    train_total_model,
    train_win_probability_model,
)
from nba_predictor.models.manifest import append_manifest_history, build_manifest, write_manifest


def _nba_season_start_year(game_date: str) -> int:
    """NBA season starting year: Oct-Dec belongs to the season starting that
    calendar year, Jan-Sep to the season that started the prior year."""
    dt = datetime.fromisoformat(str(game_date).replace("Z", "+00:00"))
    return dt.year if dt.month >= 10 else dt.year - 1


def run_retrain_pipeline(games: pd.DataFrame, models_dir: Path, model_version: str, trained_at: str) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)

    train_games, holdout_games = chronological_split(games, date_col="game_date", holdout_fraction=0.2)

    latest_season = _nba_season_start_year(games["game_date"].max())
    n_current_season_games = int(
        (train_games["game_date"].apply(_nba_season_start_year) == latest_season).sum()
    )

    train_df, feature_cols = build_training_frame(train_games)
    holdout_df, _ = build_training_frame(pd.concat([train_games, holdout_games], ignore_index=True))
    holdout_df = holdout_df[holdout_df["game_id"].isin(holdout_games["game_id"])]

    win_model = train_win_probability_model(train_df[feature_cols], train_df["home_win"])
    margin_model = train_margin_model(train_df[feature_cols], train_df["home_pts"] - train_df["away_pts"])
    total_model = train_total_model(train_df[feature_cols], train_df["home_pts"] + train_df["away_pts"])

    metrics = {}
    # Compute walk-forward metrics as required
    wf_win = None
    wf_margin = None
    wf_total = None
    try:
        wf_win = walk_forward_metrics(train_df, model_factory=lambda tr: lambda X: predict_win_probability(win_model, X[feature_cols]), windows=4)
    except Exception:
        wf_win = None
    try:
        wf_margin = walk_forward_regression(train_df, model_factory=lambda tr: lambda X: margin_model.predict(X[feature_cols]), target="home_margin", windows=4)
    except Exception:
        wf_margin = None
    try:
        total_target = train_df["home_pts"] + train_df["away_pts"]
        wf_total = walk_forward_regression(train_df.assign(home_total=total_target), model_factory=lambda tr: lambda X: total_model.predict(X[feature_cols]), target="home_total", windows=4)
    except Exception:
        wf_total = None

    if len(holdout_df) > 0:
        win_probs = predict_win_probability(win_model, holdout_df[feature_cols])
        win_preds = (win_probs >= 0.5).astype(int)
        metrics["win_probability"] = {"accuracy": float((win_preds == holdout_df["home_win"]).mean())}

        margin_preds = margin_model.predict(holdout_df[feature_cols])
        actual_margin = holdout_df["home_pts"] - holdout_df["away_pts"]
        metrics["margin"] = {"mae": float(np.mean(np.abs(margin_preds - actual_margin)))}

        total_preds = total_model.predict(holdout_df[feature_cols])
        actual_total = holdout_df["home_pts"] + holdout_df["away_pts"]
        metrics["total"] = {"mae": float(np.mean(np.abs(total_preds - actual_total)))}
    else:
        metrics = {"win_probability": {"accuracy": None}, "margin": {"mae": None}, "total": {"mae": None}}

    joblib.dump(win_model, models_dir / "win_probability_model.pkl")
    joblib.dump(margin_model, models_dir / "margin_model.pkl")
    joblib.dump(total_model, models_dir / "total_model.pkl")

    # Place probability metrics under metrics.win_probability (frontend reads per-model metrics)
    if wf_win:
        metrics.setdefault("win_probability", {})["log_loss"] = wf_win["pooled"].get("log_loss")
        metrics.setdefault("win_probability", {})["brier"] = wf_win["pooled"].get("brier")
        metrics.setdefault("win_probability", {})["auc"] = wf_win["pooled"].get("auc")
    if wf_margin:
        metrics.setdefault("margin", {})["mae"] = wf_margin["pooled"]["mae"]
        if "naive_mae_fixed" in wf_margin["pooled"]:
            metrics.setdefault("margin", {})["naive_mae_fixed"] = wf_margin["pooled"]["naive_mae_fixed"]
    if wf_total:
        metrics.setdefault("total", {})["mae"] = wf_total["pooled"]["mae"]

    manifest = build_manifest(
        model_names=["win_probability", "margin", "total"],
        metrics=metrics,
        model_version=model_version,
        trained_at=trained_at,
        training={
            "n_train_games": int(len(train_games)),
            "n_holdout_games": int(len(holdout_games)),
            "n_current_season_games": n_current_season_games,
        },
    )
    write_manifest(manifest, models_dir / "manifest.json")
    append_manifest_history(manifest, models_dir / "manifest_history.jsonl")

    return manifest
