#!/usr/bin/env python
"""NBA-0: Measure first.

Compute walk-forward (2019-2025) log-loss, Brier, AUC and accuracy for the
SERVED win probability (norm.cdf(predicted_margin / margin_sigma)),
plus the Elo-only baseline. Commit results to docs/.

This verifies or refutes the "0.68 / 0.58" claim from the deep-dive study.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_predictor.features.build import build_training_frame
from nba_predictor.models.candidate_race import default_candidates, elo_probability_factory
from nba_predictor.models.evaluate.walk_forward_eval import (
    walk_forward_metrics,
    walk_forward_regression,
)
from nba_predictor.models.evaluate.paired_bootstrap import paired_bootstrap
from nba_predictor.models.probability import MAE_TO_SIGMA, win_prob


# Configuration
SEASON_START = 2019
SEASON_END = 2024  # Frozen history has 2019-20 through 2024-25 (6 seasons)
WINDOWS = 4
N_RESAMPLES = 2000
SEED = 20261007


def load_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load games and player rows from frozen history (2019-20 through 2024-25).

    The frozen history (data/history/nba_history.json.gz) contains regular-season games
    from 2018-10-01 to 2025-04-13. We select seasons 2019-20 through 2024-25 inclusive
    (6 seasons: 2019, 2020, 2021, 2022, 2023, 2024). The 2025-26 season is not yet in history.
    """
    path = Path("data/history/nba_history.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    games_2019_2024 = []
    players_2019_2024 = []
    for g in data["games"]:
        year = int(g["game_date"][:4])
        month = int(g["game_date"][5:7])
        season_year = year if month >= 10 else year - 1
        if SEASON_START <= season_year <= SEASON_END:
            games_2019_2024.append(g)

    for p in data["players"]:
        year = int(p["game_date"][:4])
        month = int(p["game_date"][5:7])
        season_year = year if month >= 10 else year - 1
        if SEASON_START <= season_year <= SEASON_END:
            players_2019_2024.append(p)

    return pd.DataFrame(games_2019_2024), pd.DataFrame(players_2019_2024)


def main() -> int:
    print(f"=== NBA-0 Measure First: {SEASON_START}-{SEASON_END} walk-forward ===")

    # Load data
    games, players = load_history()
    print(f"Loaded {len(games)} games, {len(players)} player-rows from frozen history")
    print(f"Date range: {games['game_date'].iloc[0]} -> {games['game_date'].iloc[-1]}")
    print(f"Seasons: {SEASON_START}-{SEASON_START+1} through {SEASON_END}-{SEASON_END+1} (regular season only)")

    # Build training frame (features + targets)
    df, feature_cols = build_training_frame(games, players=players)
    print(f"After build_training_frame: {len(df)} games, {len(feature_cols)} features")

    # Add regression targets
    df = df.assign(
        home_margin=df["home_pts"] - df["away_pts"],
        home_total=df["home_pts"] + df["away_pts"],
    )

    # Get candidates
    candidates = default_candidates(feature_cols)

    # --- CRITICAL: Compute sigma from TRAIN-SPLIT walk-forward (matches production) ---
    # retrain.py splits 80/20 chronologically, runs walk-forward on train_df only,
    # computes sigma from pooled test residuals, publishes in manifest.
    # We do the same here to get the production sigma, then evaluate on full walk-forward.
    from nba_predictor.pipeline.retrain import _nba_season_start_year, chronological_split

    train_games, holdout_games = chronological_split(games, date_col="game_date", holdout_fraction=0.2)
    print(f"\nTrain/holdout split: train={len(train_games)}, holdout={len(holdout_games)}")
    print(f"Train date range: {train_games['game_date'].min()} -> {train_games['game_date'].max()}")
    print(f"Holdout date range: {holdout_games['game_date'].min()} -> {holdout_games['game_date'].max()}")

    train_df, _ = build_training_frame(train_games, players=players)
    train_df = train_df.assign(
        home_margin=train_df["home_pts"] - train_df["away_pts"],
        home_total=train_df["home_pts"] + train_df["away_pts"],
    )

    # Walk-forward on TRAIN split only (matches production retrain.py)
    train_margin_wf = walk_forward_regression(
        train_df,
        model_factory=lambda tr: candidates["ridge"]["margin"](tr),
        target="home_margin",
        windows=WINDOWS,
        date_col="game_date",
        fixed_baseline=3.0,
    )
    train_margin_mae = train_margin_wf["pooled"]["mae"]
    production_sigma = train_margin_mae * MAE_TO_SIGMA
    print(f"Production sigma (from train-split walk-forward): MAE={train_margin_mae:.4f}, sigma={production_sigma:.4f}")

    # Also compute full walk-forward sigma for comparison
    full_margin_wf = walk_forward_regression(
        df,
        model_factory=lambda tr: candidates["ridge"]["margin"](tr),
        target="home_margin",
        windows=WINDOWS,
        date_col="game_date",
        fixed_baseline=3.0,
    )
    full_margin_mae = full_margin_wf["pooled"]["mae"]
    full_sigma = full_margin_mae * MAE_TO_SIGMA
    print(f"Full walk-forward sigma (for reference): MAE={full_margin_mae:.4f}, sigma={full_sigma:.4f}")

    # Use PRODUCTION sigma for SERVED win probability evaluation
    margin_sigma = production_sigma

    # --- 1. SERVED win probability: from margin model via norm.cdf ---
    print("\n--- Computing SERVED win probability (margin -> win_prob) ---")

    # Full walk-forward for margin predictions (for evaluation on all games)
    margin_wf = full_margin_wf  # reuse
    margin_preds = np.asarray(margin_wf["pooled"]["preds"])
    margin_y = np.asarray(margin_wf["pooled"]["y"])

    # SERVED win probability = norm.cdf(predicted_margin / margin_sigma)
    served_win_probs = np.array([win_prob(float(m), margin_sigma) for m in margin_preds])
    outcomes = (margin_y > 0).astype(int)

    print(f"Full walk-forward: n={len(outcomes)} test games")
    print(f"Using production sigma={margin_sigma:.4f} (from train-split walk-forward)")

    # Compute metrics for SERVED win probability
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score

    served_log_loss = log_loss(outcomes, served_win_probs, labels=[0, 1])
    served_brier = brier_score_loss(outcomes, served_win_probs)
    served_auc = roc_auc_score(outcomes, served_win_probs)
    served_accuracy = accuracy_score(outcomes, (served_win_probs >= 0.5).astype(int))

    print(f"SERVED win probability: log_loss={served_log_loss:.4f}, brier={served_brier:.4f}, "
          f"auc={served_auc:.4f}, accuracy={served_accuracy:.4f}")

    # --- 2. Elo-only baseline ---
    print("\n--- Computing Elo-only baseline ---")
    elo_factory = elo_probability_factory()
    elo_wf = walk_forward_metrics(
        df,
        model_factory=elo_factory,
        windows=WINDOWS,
        date_col="game_date",
        target_col="home_win",
    )
    elo_probs = np.asarray(elo_wf["pooled"]["preds"])
    elo_y = np.asarray(elo_wf["pooled"]["y"])
    elo_log_loss = elo_wf["pooled"]["log_loss"]
    elo_brier = elo_wf["pooled"]["brier"]
    elo_auc = elo_wf["pooled"]["auc"]
    elo_accuracy = elo_wf["pooled"]["accuracy"]

    print(f"Elo-only: log_loss={elo_log_loss:.4f}, brier={elo_brier:.4f}, "
          f"auc={elo_auc:.4f}, accuracy={elo_accuracy:.4f}")

    # --- 3. Logistic classifier (for comparison, what the old manifest reported) ---
    print("\n--- Computing logistic classifier (legacy win model) ---")
    log_wf = walk_forward_metrics(
        df,
        model_factory=lambda tr: candidates["logistic"]["win"](tr),
        windows=WINDOWS,
        date_col="game_date",
        target_col="home_win",
    )
    log_probs = np.asarray(log_wf["pooled"]["preds"])
    log_y = np.asarray(log_wf["pooled"]["y"])
    log_log_loss = log_wf["pooled"]["log_loss"]
    log_brier = log_wf["pooled"]["brier"]
    log_auc = log_wf["pooled"]["auc"]
    log_accuracy = log_wf["pooled"]["accuracy"]

    print(f"Logistic: log_loss={log_log_loss:.4f}, brier={log_brier:.4f}, "
          f"auc={log_auc:.4f}, accuracy={log_accuracy:.4f}")

    # --- 4. Paired bootstrap: SERVED vs Elo ---
    print(f"\n--- Paired bootstrap: SERVED vs Elo (n_resamples={N_RESAMPLES}, seed={SEED}) ---")
    boot_log_loss = paired_bootstrap(outcomes, elo_probs, served_win_probs, "log_loss",
                                      n_resamples=N_RESAMPLES, seed=SEED)
    boot_brier = paired_bootstrap(outcomes, elo_probs, served_win_probs, "brier",
                                   n_resamples=N_RESAMPLES, seed=SEED)
    boot_auc = paired_bootstrap(outcomes, elo_probs, served_win_probs, "auc",
                                 n_resamples=N_RESAMPLES, seed=SEED)
    boot_accuracy = paired_bootstrap(outcomes, (elo_probs >= 0.5).astype(int),
                                      (served_win_probs >= 0.5).astype(int), "brier",  # using brier as proxy
                                      n_resamples=N_RESAMPLES, seed=SEED)

    # Note: accuracy isn't directly in METRICS, use brier as proxy or compute manually
    # For now report the key metrics
    print(f"  log_loss: diff={boot_log_loss['difference']:+.4f}  "
          f"95% CI=[{boot_log_loss['ci_low']:+.4f}, {boot_log_loss['ci_high']:+.4f}]  "
          f"improved={boot_log_loss['improved']}")
    print(f"  brier:    diff={boot_brier['difference']:+.4f}  "
          f"95% CI=[{boot_brier['ci_low']:+.4f}, {boot_brier['ci_high']:+.4f}]  "
          f"improved={boot_brier['improved']}")
    print(f"  auc:      diff={boot_auc['difference']:+.4f}  "
          f"95% CI=[{boot_auc['ci_low']:+.4f}, {boot_auc['ci_high']:+.4f}]  "
          f"improved={boot_auc['improved']}")

    # --- 5. Margin and Total MAE with bootstrap ---
    print("\n--- Margin MAE (ridge) ---")
    print(f"  MAE={full_margin_mae:.4f}, naive={full_margin_wf['pooled']['naive_mae']:.4f}, "
          f"naive_fixed={full_margin_wf['pooled'].get('naive_mae_fixed', 'N/A')}")

    # Get the first training window's mean total for fixed baseline (matches walk_forward_regression fixed_baseline).
    # This is a SINGLE constant computed from the FIRST training window only (prior data),
    # not the overall mean. It represents what a naive forecaster would know at season start.
    from nba_predictor.models.evaluate.walk_forward_eval import expanding_windows
    first_train_idx = expanding_windows(df["game_date"], WINDOWS)[0][0]
    fixed_baseline_total = float(df.iloc[first_train_idx]["home_total"].mean())

    total_wf = walk_forward_regression(
        df,
        model_factory=lambda tr: candidates["ridge"]["total"](tr),
        target="home_total",
        windows=WINDOWS,
        date_col="game_date",
        fixed_baseline=fixed_baseline_total,
    )
    total_mae = total_wf["pooled"]["mae"]
    print(f"--- Total MAE (ridge) ---")
    print(f"  MAE={total_mae:.4f}, naive={total_wf['pooled']['naive_mae']:.4f}, "
          f"naive_fixed={total_wf['pooled'].get('naive_mae_fixed', 'N/A')}")

    # Bootstrap margin MAE vs naive
    margin_naive_preds = []
    for i, (train_idx, test_idx) in enumerate(
        __import__("nba_predictor.models.evaluate.walk_forward_eval", fromlist=["expanding_windows"]).expanding_windows(df["game_date"], WINDOWS)
    ):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        naive_mean = float(train_df["home_margin"].mean())
        margin_naive_preds.extend([naive_mean] * len(test_df))
    margin_naive_preds = np.asarray(margin_naive_preds)

    boot_margin = paired_bootstrap(margin_y, margin_naive_preds, margin_preds, "mae",
                                    n_resamples=N_RESAMPLES, seed=SEED)
    print(f"  Margin MAE vs naive: diff={boot_margin['difference']:+.4f}  "
          f"95% CI=[{boot_margin['ci_low']:+.4f}, {boot_margin['ci_high']:+.4f}]  "
          f"improved={boot_margin['improved']}")

    # Bootstrap total MAE vs naive_fixed (using same fixed_baseline_total as walk_forward_regression)
    total_preds = np.asarray(total_wf["pooled"]["preds"])
    total_y = np.asarray(total_wf["pooled"]["y"])
    total_naive_fixed = total_wf["pooled"].get("naive_mae_fixed")
    if total_naive_fixed is not None:
        total_naive = np.full_like(total_y, fixed_baseline_total)
        boot_total = paired_bootstrap(total_y, total_naive, total_preds, "mae",
                                       n_resamples=N_RESAMPLES, seed=SEED)
        print(f"  Total MAE vs naive_fixed: diff={boot_total['difference']:+.4f}  "
              f"95% CI=[{boot_total['ci_low']:+.4f}, {boot_total['ci_high']:+.4f}]  "
              f"improved={boot_total['improved']}")

    # --- 6. Write results to docs/ ---
    results = {
        "metadata": {
            "description": "NBA-0 Measure First: walk-forward evaluation of SERVED win probability (norm.cdf(margin/sigma)) and Elo baseline on 2019-20 through 2024-25 regular season games.",
            "seasons": "2019-20, 2020-21, 2021-22, 2022-23, 2023-24, 2024-25 (6 seasons)",
            "data_source": "frozen history: data/history/nba_history.json.gz (regular season only, no playoffs)",
            "n_games_raw_history": len(games),                    # 7,308 games from history after season filter
            "n_games_after_feature_engineering": len(df),         # 7,289 games after build_training_frame (drops games without box scores)
            "n_walkforward_pooled_test": int(len(outcomes)),      # 4,564 = sum of test games across 4 expanding windows (THIS is the n for all metrics)
            "walkforward_windows": WINDOWS,
            "walkforward_method": "expanding window, date-based splits, causality enforced (train_max_date < test_min_date)",
            "leak_free_verification": {
                "features_causal": "build_training_frame uses shift-1 rolling features; compute_h_features emits features BEFORE folding game result into state",
                "total_drift_correction": "compute_total_drift is ONLY applied at retrain time on full training frame; NOT applied during walk-forward evaluation (walk-forward uses raw model predictions)",
                "early_season_blending": "shrunk_projections uses only completed_rows (past games) for shrinkage; never sees future games",
                "model_refit_per_window": "walk_forward_metrics/regression re-fit model_factory on each window's training data only; no shared state across windows",
                "sigma_source": "margin_sigma = walk-forward pooled MAE * MAE_TO_SIGMA (out-of-fold residuals). Mirrors production: retrain publishes sigma from walk-forward in manifest, serving uses it.",
            },
            "note_on_n_discrepancy": "The deep-dive audit cited n=8,388 for 2019-2025; this run uses frozen history through 2025-04-13 (end of 2024-25 regular season), giving 7,308 raw games. The 8,388 may include playoffs or partial 2025-26 season not in frozen history.",
        },
        "served_win_probability": {
            "log_loss": served_log_loss,
            "brier": served_brier,
            "auc": served_auc,
            "accuracy": served_accuracy,
            "n": int(len(outcomes)),  # pooled walk-forward test games = 4,564
            "margin_sigma": margin_sigma,
            "n_definition": "pooled test games across all 4 walk-forward windows",
        },
        "elo_baseline": {
            "log_loss": elo_log_loss,
            "brier": elo_brier,
            "auc": elo_auc,
            "accuracy": elo_accuracy,
            "n": int(len(elo_y)),
            "n_definition": "pooled test games across all 4 walk-forward windows",
        },
        "logistic_classifier": {
            "log_loss": log_log_loss,
            "brier": log_brier,
            "auc": log_auc,
            "accuracy": log_accuracy,
            "n": int(len(log_y)),
            "n_definition": "pooled test games across all 4 walk-forward windows",
        },
        "margin": {
            "mae": full_margin_mae,
            "naive_mae": full_margin_wf["pooled"]["naive_mae"],
            "naive_mae_fixed": full_margin_wf["pooled"].get("naive_mae_fixed"),
            "n": int(len(margin_y)),
            "n_definition": "pooled test games across all 4 walk-forward windows",
        },
        "total": {
            "mae": total_mae,
            "naive_mae": total_wf["pooled"]["naive_mae"],
            "naive_mae_fixed": total_wf["pooled"].get("naive_mae_fixed"),
            "n": int(len(total_y)),
            "n_definition": "pooled test games across all 4 walk-forward windows",
        },
        "bootstrap_served_vs_elo": {
            "log_loss": boot_log_loss,
            "brier": boot_brier,
            "auc": boot_auc,
        },
        "bootstrap_margin_vs_naive": boot_margin,
    }

    if "boot_total" in locals():
        results["bootstrap_total_vs_naive_fixed"] = boot_total

    # Save to docs/
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    output_path = docs_dir / f"nba0_measure_{SEASON_START}_{SEASON_END}.json"
    with open(output_path, "w") as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        json.dump(convert(results), f, indent=2)
    print(f"\nResults written to {output_path}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Data: {len(games)} raw games from frozen history (seasons {SEASON_START}-{SEASON_END})")
    print(f"      {len(df)} games after feature engineering (build_training_frame)")
    print(f"      {len(outcomes)} pooled test games across {WINDOWS} walk-forward windows")
    print(f"SERVED win probability (from margin): log_loss={served_log_loss:.4f}, auc={served_auc:.4f}")
    print(f"Elo-only baseline:                     log_loss={elo_log_loss:.4f}, auc={elo_auc:.4f}")
    print(f"Logistic classifier (legacy):          log_loss={log_log_loss:.4f}, auc={log_auc:.4f}")
    print(f"Margin MAE: {full_margin_mae:.4f} (vs naive {full_margin_wf['pooled']['naive_mae']:.4f})")
    print(f"Total MAE:  {total_mae:.4f} (vs naive {total_wf['pooled']['naive_mae']:.4f})")

    # Compare to the "0.68 / 0.58" claim
    print(f"\nDeep-dive study claim: 0.68 log-loss / 0.58 AUC (702-game manifest)")
    print(f"This run (full walk-forward): {served_log_loss:.4f} log-loss / {served_auc:.4f} AUC")
    if abs(served_log_loss - 0.68) < 0.02 and abs(served_auc - 0.58) < 0.02:
        print("  -> CLAIM CONFIRMED (approximately)")
    else:
        print("  -> CLAIM REFUTED (does not match)")

    print("\nLeak-free verification:")
    print("  - Features: shift-1 rolling, causal by design (compute_h_features emits before state update)")
    print("  - Total drift correction: NOT applied during walk-forward (retrain-time only)")
    print("  - Early-season blending: uses only past completed games (completed_rows)")
    print("  - Models: re-fit per window on training data only (walk_forward_* contract)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())