#!/usr/bin/env python
"""NBA-1: pre-game Elo ratings as features, walk-forward A/B on the SERVED win probability.

Arms (identical games, identical 4 expanding windows as NBA-0):
  baseline  = production feature set (FEATURE_COLUMNS), StandardScaler + Ridge margin model
  candidate = baseline + pregame_elo_diff_s (each team's rating as of tip-off)

Served win probability = norm.cdf(predicted_margin / sigma), sigma taken from the
FIRST window's training residuals only (same rule as tools/nba0_measure.py).

Per window, K and home-court advantage are chosen by `select_elo_params` on that
window's training data only. The rating for a test game still uses only games
dated before it, which is exactly what is knowable at tip-off.

Ship rule: paired bootstrap on per-game log-loss difference, 95% CI entirely
below zero; AUC, Brier and calibration gap not worse than baseline.

Run (from repo root):
    PYTHONPATH=src python tools/nba1_pregame_elo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent / "src"))

from nba0_measure import SEASON_END, SEASON_START, WINDOWS, load_history  # noqa: E402

from nba_predictor.features.build import build_training_frame  # noqa: E402
from nba_predictor.features.pregame_elo import (  # noqa: E402
    PREGAME_ELO_FEATURE,
    pregame_elo_ratings,
    select_elo_params,
)
from nba_predictor.models.artifact_gate import calibration_gap  # noqa: E402
from nba_predictor.models.evaluate.paired_bootstrap import paired_bootstrap  # noqa: E402
from nba_predictor.models.evaluate.walk_forward_eval import expanding_windows  # noqa: E402
from nba_predictor.models.probability import MAE_TO_SIGMA  # noqa: E402

N_RESAMPLES = 2000
SEED = 20261007
OUT_JSON = Path("docs") / "nba1_pregame_elo_2019_2024.json"


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error, equal-width bins, weighted by bin count."""
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(total / len(y))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
        "ece_10bin": ece(y, p),
        "calibration_gap_5bucket": float(calibration_gap(y, p, n_buckets=5)),
    }


def ridge_margin_factory(feature_cols: list[str]):
    def fit(train: pd.DataFrame):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(train[feature_cols], train["home_margin"])
        return model
    return fit


def run_arm(frames: list[pd.DataFrame], windows, feature_cols: list[str]):
    """Walk-forward over `windows`; `frames[i]` carries window i's feature columns.

    Sigma is the window-0 model's in-sample MAE times MAE_TO_SIGMA, fitted before
    any window is scored and then held constant for every window (same rule as
    tools/nba0_measure.py). Returns served probs, outcomes, game_ids, window info, sigma.
    """
    pooled_p, pooled_y, pooled_id, per_window = [], [], [], []
    sigma = None
    for i, (train_idx, test_idx) in enumerate(windows):
        frame = frames[i]
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(train[feature_cols], train["home_margin"])
        if sigma is None:
            resid = np.abs(train["home_margin"].to_numpy() - model.predict(train[feature_cols]))
            sigma = float(np.mean(resid) * MAE_TO_SIGMA)
        pred = model.predict(test[feature_cols])
        pooled_p.append(norm.cdf(pred / sigma))
        pooled_y.append((test["home_margin"].to_numpy() > 0).astype(int))
        pooled_id.append(test["game_id"].to_numpy())
        per_window.append({"window": i, "n_train": int(len(train)), "n_test": int(len(test)),
                           "train_max": str(train["game_date"].max()),
                           "test_min": str(test["game_date"].min())})
    return (np.concatenate(pooled_p), np.concatenate(pooled_y), np.concatenate(pooled_id),
            per_window, sigma)


def main() -> int:
    t0 = time.time()
    print(f"=== NBA-1 pre-game Elo A/B: {SEASON_START}-{SEASON_END}, {WINDOWS} walk-forward windows ===")

    games, players = load_history()
    df, feature_cols = build_training_frame(games, players=players)
    df = df.sort_values("game_date").reset_index(drop=True)
    df = df.assign(home_margin=df["home_pts"] - df["away_pts"])
    print(f"Frame: {len(df)} games, {len(feature_cols)} production features")

    # Elo runs over every completed game in history, in date order. A row's rating only
    # ever reads games on earlier dates, so this is causal for every row.
    elo_games = games[["game_id", "game_date", "home_team", "away_team", "home_win"]].copy()
    elo_games["home_win"] = elo_games["home_win"].astype(float)

    windows = expanding_windows(df["game_date"], WINDOWS)

    # Per-window parameter selection, then the rating column built with that window's params.
    selected = []
    rating_cols = []
    for i, (train_idx, _) in enumerate(windows):
        train_ids = set(df.iloc[train_idx]["game_id"])
        train_elo = elo_games[elo_games["game_id"].isin(train_ids)]
        k, hca = select_elo_params(train_elo)
        selected.append({"window": i, "k": k, "home_court_advantage": hca})
        print(f"  window {i}: selected K={k:g}, HCA={hca:g} (train-only)")
        ratings = pregame_elo_ratings(elo_games, k=k, hca=hca).set_index("game_id")
        r = ratings.loc[df["game_id"]]
        rating_cols.append(pd.Series((r["home_pregame_elo"].to_numpy() - r["away_pregame_elo"].to_numpy()) / 100,
                                     index=df.index))

    base_cols = feature_cols
    cand_cols = feature_cols + [PREGAME_ELO_FEATURE]
    frames = [df.assign(**{PREGAME_ELO_FEATURE: rating_cols[i].to_numpy()}) for i in range(len(windows))]

    base_p, y, base_id, base_info, sigma_b = run_arm(frames, windows, base_cols)
    cand_p, y_c, cand_id, cand_info, sigma_c = run_arm(frames, windows, cand_cols)
    assert np.array_equal(base_id, cand_id), "arms scored different games"
    assert np.array_equal(y, y_c)

    m_base = metrics(y, base_p)
    m_cand = metrics(y, cand_p)
    print(f"\nBaseline  (production features): {json.dumps({k: round(v, 4) for k, v in m_base.items()})}")
    print(f"Candidate (+ pre-game Elo):      {json.dumps({k: round(v, 4) for k, v in m_cand.items()})}")

    boots = {}
    for metric in ("log_loss", "brier", "auc"):
        boots[metric] = paired_bootstrap(y, base_p, cand_p, metric, n_resamples=N_RESAMPLES, seed=SEED)
        b = boots[metric]
        print(f"  {metric:9s} diff={b['difference']:+.5f}  95% CI=[{b['ci_low']:+.5f}, {b['ci_high']:+.5f}]  "
              f"improved={b['improved']}")

    ll_passes = boots["log_loss"]["improved"]
    auc_ok = m_cand["auc"] >= m_base["auc"]
    brier_ok = m_cand["brier"] <= m_base["brier"]
    ece_ok = m_cand["ece_10bin"] <= m_base["ece_10bin"]
    cal_ok = m_cand["calibration_gap_5bucket"] <= m_base["calibration_gap_5bucket"]
    ship = bool(ll_passes and auc_ok and brier_ok and ece_ok and cal_ok)
    print(f"\nGate: log-loss CI below zero={ll_passes}, AUC not worse={auc_ok}, "
          f"Brier not worse={brier_ok}, ECE not worse={ece_ok}, calibration gap not worse={cal_ok}")
    print(f"SHIP={ship}")

    runtime = time.time() - t0
    result = {
        "metadata": {
            "description": "NBA-1 pre-game Elo features: walk-forward A/B on the served win probability.",
            "seasons": f"{SEASON_START}-{SEASON_START+1} through {SEASON_END}-{SEASON_END+1}",
            "windows": WINDOWS,
            "n_pooled_test": int(len(y)),
            "n_resamples": N_RESAMPLES,
            "seed": SEED,
            "runtime_seconds": round(runtime, 1),
            "elo_scale": 400,
            "season_regression": "1/3 toward 1500 at each team's first game of a new season",
            "param_grid": {"K": [10, 20, 30], "home_court_advantage": [0, 50, 100]},
            "sigma": {"baseline": sigma_b, "candidate": sigma_c},
            "leak_control": "rating for a game reads only games on earlier dates; K/HCA chosen on each window's training data",
        },
        "selected_params": selected,
        "windows": base_info,
        "baseline": m_base,
        "candidate": m_cand,
        "bootstrap": boots,
        "gate": {"log_loss_ci_below_zero": ll_passes, "auc_not_worse": auc_ok,
                 "brier_not_worse": brier_ok, "ece_not_worse": ece_ok,
                 "calibration_gap_not_worse": cal_ok, "ship": ship},
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nWritten {OUT_JSON}  (runtime {runtime:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
