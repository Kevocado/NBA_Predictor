from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_predictor.features.build import build_training_frame
from nba_predictor.models.evaluate.walk_forward import chronological_split
from nba_predictor.models.game_outcome import (
    predict_win_probability,
    train_margin_model,
    train_total_model,
    train_win_probability_model,
)
from nba_predictor.models.manifest import append_manifest_history, build_manifest, write_manifest


def run_retrain_pipeline(games: pd.DataFrame, models_dir: Path, model_version: str, trained_at: str) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)

    train_games, holdout_games = chronological_split(games, date_col="game_date", holdout_fraction=0.2)

    train_df, feature_cols = build_training_frame(train_games)
    holdout_df, _ = build_training_frame(pd.concat([train_games, holdout_games], ignore_index=True))
    holdout_df = holdout_df[holdout_df["game_id"].isin(holdout_games["game_id"])]

    win_model = train_win_probability_model(train_df[feature_cols], train_df["home_win"])
    margin_model = train_margin_model(train_df[feature_cols], train_df["home_pts"] - train_df["away_pts"])
    total_model = train_total_model(train_df[feature_cols], train_df["home_pts"] + train_df["away_pts"])

    metrics = {}
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

    manifest = build_manifest(
        model_names=["win_probability", "margin", "total"],
        metrics=metrics,
        model_version=model_version,
        trained_at=trained_at,
    )
    write_manifest(manifest, models_dir / "manifest.json")
    append_manifest_history(manifest, models_dir / "manifest_history.jsonl")

    return manifest
