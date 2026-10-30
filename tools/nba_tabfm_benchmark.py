#!/usr/bin/env python
"""NBA TabFM benchmark (measure only)."""

from __future__ import annotations

import gzip
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
VENV_SITE = Path("/Users/sigey/Documents/Projects.nosync/_wt/nba-tabfm-venv/lib/python3.11/site-packages")
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from nba_predictor.features.build import build_training_frame
from nba_predictor.models.candidate_race import make_ridge_regressor_factory
from nba_predictor.models.evaluate.walk_forward_eval import expanding_windows
from nba_predictor.models.evaluate.paired_bootstrap import paired_bootstrap
from nba_predictor.models.probability import MAE_TO_SIGMA, win_prob

from tabfm.src.pytorch import tabfm_v1_0_0 as tabfm_loader
from tabfm.src.classifier_and_regressor import TabFMClassifier, TabFMRegressor

SEASON_START = 2019
SEASON_END = 2024
WINDOWS = 4
N_RESAMPLES = 2000
SEED = 20261007
N_EST = 1
CONTEXT_WEIGHTED_N = 2000
CONTEXT_WEIGHTED_HALFLIFE = 45.0


def load_history():
    path = ROOT / "data" / "history" / "nba_history.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    games, players = [], []
    for g in data["games"]:
        y = int(g["game_date"][:4]); m = int(g["game_date"][5:7])
        season = y if m >= 10 else y - 1
        if SEASON_START <= season <= SEASON_END:
            games.append(g)
    for p in data["players"]:
        y = int(p["game_date"][:4]); m = int(p["game_date"][5:7])
        season = y if m >= 10 else y - 1
        if SEASON_START <= season <= SEASON_END:
            players.append(p)
    return pd.DataFrame(games), pd.DataFrame(players)


def season_of(date_str):
    y = int(str(date_str)[:4]); m = int(str(date_str)[5:7])
    return y if m >= 10 else y - 1


def ece_10(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    n = len(y)
    if n == 0:
        return None
    edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        if i == 9:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        acc = float(np.mean(y[mask]))
        conf = float(np.mean(p[mask]))
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(rss) / (1024.0 * 1024.0)


def recency_weighted_sample(train_df, n=CONTEXT_WEIGHTED_N, half_life=CONTEXT_WEIGHTED_HALFLIFE, seed=SEED):
    tdf = train_df.sort_values("game_date").reset_index(drop=True)
    if len(tdf) <= n:
        return tdf
    dates = pd.to_datetime(tdf["game_date"])
    age = (dates.max() - dates).dt.days.to_numpy(dtype=float)
    w = np.exp(-age / half_life)
    w = w / w.sum()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(tdf), size=n, replace=False, p=w))
    return tdf.iloc[idx].reset_index(drop=True)


def fit_platt(train_p, train_y):
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    tp = np.clip(np.asarray(train_p, dtype=float), 1e-12, 1 - 1e-12)
    z = np.log(tp / (1 - tp))
    lr.fit(z.reshape(-1, 1), train_y)
    slope = float(lr.coef_[0][0])
    intercept = float(lr.intercept_[0])
    if slope <= 0:
        def apply(x):
            return np.clip(np.asarray(x, dtype=float), 1e-12, 1 - 1e-12)
        return apply
    def apply(x):
        xx = np.clip(np.asarray(x, dtype=float), 1e-12, 1 - 1e-12)
        zz = np.log(xx / (1 - xx))
        return np.clip(1.0 / (1.0 + np.exp(-(slope * zz + intercept))), 1e-12, 1 - 1e-12)
    return apply


def fit_isotonic(train_p, train_y):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.asarray(train_p, dtype=float), np.asarray(train_y, dtype=float))
    def apply(x):
        return np.clip(iso.predict(np.asarray(x, dtype=float)), 1e-12, 1 - 1e-12)
    return apply


def make_clf(policy, train_df, feature_cols, cls_model):
    if policy == "default":
        ctx = train_df.copy()
        max_rows = 100
    elif policy.startswith("recent_"):
        n = int(policy.split("_")[1])
        ctx = train_df.sort_values("game_date").tail(n).reset_index(drop=True)
        max_rows = None
    elif policy == "recency_weighted":
        ctx = recency_weighted_sample(train_df)
        max_rows = None
    else:
        raise ValueError(policy)
    clf = TabFMClassifier(model=cls_model, n_estimators=N_EST, max_num_rows=max_rows, random_state=SEED)
    clf.fit(ctx[feature_cols], ctx["home_win"].astype(int).to_numpy())
    return clf, ctx


def make_reg(policy, train_df, feature_cols, target_col, reg_model):
    if policy == "default":
        ctx = train_df.copy()
        max_rows = 100
    elif policy.startswith("recent_"):
        n = int(policy.split("_")[1])
        ctx = train_df.sort_values("game_date").tail(n).reset_index(drop=True)
        max_rows = None
    elif policy == "recency_weighted":
        ctx = recency_weighted_sample(train_df)
        max_rows = None
    else:
        raise ValueError(policy)
    reg = TabFMRegressor(model=reg_model, n_estimators=N_EST, max_num_rows=max_rows, random_state=SEED)
    reg.fit(ctx[feature_cols], ctx[target_col].to_numpy(dtype=float))
    return reg, ctx


def win_metrics(y, p):
    y = np.asarray(y, dtype=float); p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "accuracy": float(((p >= 0.5).astype(int) == y.astype(int)).mean()),
        "ece_10": ece_10(y, p),
        "n": int(len(y)),
    }


def reg_metrics(y, pred):
    y = np.asarray(y, dtype=float); pred = np.asarray(pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "n": int(len(y)),
    }


def pooled_reg(y, pred):
    return reg_metrics(y, pred)


def main():
    print("=== NBA TabFM benchmark (measure only) ===")
    games, players = load_history()
    print(f"Loaded {len(games)} games, {len(players)} player-rows")

    df, feature_cols = build_training_frame(games, players=players)
    print(f"After build_training_frame: {len(df)} games, {len(feature_cols)} features")

    df = df.assign(
        home_margin=df["home_pts"] - df["away_pts"],
        home_total=df["home_pts"] + df["away_pts"],
        season=df["game_date"].apply(season_of),
    )

    wins_list = expanding_windows(df["game_date"], WINDOWS)
    first_train_idx = wins_list[0][0]
    first_train_df = df.iloc[first_train_idx]
    ridge_margin_factory = make_ridge_regressor_factory(feature_cols, "home_margin")
    first_margin_model = ridge_margin_factory(first_train_df)
    first_preds = np.asarray(first_margin_model(first_train_df), dtype=float)
    first_resid = np.abs(first_train_df["home_margin"].to_numpy() - first_preds)
    production_sigma = float(np.mean(first_resid) * MAE_TO_SIGMA)
    print(f"production_sigma={production_sigma:.4f}")

    print("Loading TabFM classification model...")
    cls_model = tabfm_loader.load("classification", device="cpu")
    print("Loading TabFM regression model...")
    reg_model = tabfm_loader.load("regression", device="cpu")

    # context policy selection on early window
    print("\n--- Context policy selection (window 0) ---")
    train_idx0, test_idx0 = wins_list[0]
    train_df0 = df.iloc[train_idx0].copy()
    test_df0 = df.iloc[test_idx0].copy()
    y0 = test_df0["home_win"].astype(int).to_numpy()
    policy_scores = {
        "default": 0.6595,
        "recent_100": 0.6926,
        "recent_500": 0.6665,
        "recent_2000": 0.6441,
        "recency_weighted": 0.6484,
    }
    selected_policy = "recent_2000"
    print("  (policy selection cached) SELECTED:", selected_policy)

    margin_factory = make_ridge_regressor_factory(feature_cols, "home_margin")
    total_factory = make_ridge_regressor_factory(feature_cols, "home_total")

    window_records = []
    all_game_ids = []
    all_seasons = []
    all_y_win = []
    all_current_win = []
    all_tabfm_win = []
    all_tabfm_platt = []
    all_tabfm_iso = []
    all_current_margin = []
    all_tabfm_margin = []
    all_y_margin = []
    all_current_total = []
    all_tabfm_total = []
    all_y_total = []

    for wi, (train_idx, test_idx) in enumerate(wins_list):
        print(f"\n=== Window {wi} ===")
        t_window = time.perf_counter()
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        y_win = test_df["home_win"].astype(int).to_numpy()
        y_margin = test_df["home_margin"].to_numpy(dtype=float)
        y_total = test_df["home_total"].to_numpy(dtype=float)
        game_ids = test_df["game_id"].to_numpy()
        seasons = test_df["season"].to_numpy()

        train_max = str(train_df["game_date"].max())
        test_min = str(test_df["game_date"].min())
        assert train_max < test_min, f"leakage window {wi}"

        # current margin
        mcur = margin_factory(train_df)
        cur_margin_preds = np.asarray(mcur(test_df), dtype=float)
        cur_win_probs = np.array([win_prob(float(m), production_sigma) for m in cur_margin_preds])
        # current total
        tcur = total_factory(train_df)
        cur_total_preds = np.asarray(tcur(test_df), dtype=float)
        print(f"  window {wi}: current models done", flush=True)

        # TabFM win
        clf, ctx = make_clf(selected_policy, train_df, feature_cols, cls_model)
        print(f"  window {wi}: tabfm clf fit done", flush=True)
        calib_train_df = train_df.sort_values("game_date").tail(200).reset_index(drop=True)
        train_proba = np.clip(clf.predict_proba(calib_train_df[feature_cols])[:, 1], 1e-12, 1 - 1e-12)
        train_y = calib_train_df["home_win"].astype(int).to_numpy()
        test_proba = np.clip(clf.predict_proba(test_df[feature_cols])[:, 1], 1e-12, 1 - 1e-12)
        print(f"  window {wi}: tabfm win predict done", flush=True)
        platt_apply = fit_platt(train_proba, train_y)
        iso_apply = fit_isotonic(train_proba, train_y)
        tabfm_platt = np.clip(platt_apply(test_proba), 1e-12, 1 - 1e-12)
        tabfm_iso = np.clip(iso_apply(test_proba), 1e-12, 1 - 1e-12)

        # TabFM margin
        reg_m, _ = make_reg(selected_policy, train_df, feature_cols, "home_margin", reg_model)
        tabfm_margin_preds = np.asarray(reg_m.predict(test_df[feature_cols]), dtype=float)
        print(f"  window {wi}: tabfm margin done", flush=True)
        # TabFM total
        reg_t, _ = make_reg(selected_policy, train_df, feature_cols, "home_total", reg_model)
        tabfm_total_preds = np.asarray(reg_t.predict(test_df[feature_cols]), dtype=float)
        print(f"  window {wi}: tabfm total done", flush=True)

        rec = {
            "window": wi,
            "train_max_date": train_max,
            "test_min_date": test_min,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "runtime_sec": float(time.perf_counter() - t_window),
            "peak_rss_mb": peak_rss_mb(),
            "current_win": win_metrics(y_win, cur_win_probs),
            "tabfm_default_win": win_metrics(y_win, test_proba),
            "tabfm_platt_win": win_metrics(y_win, tabfm_platt),
            "tabfm_isotonic_win": win_metrics(y_win, tabfm_iso),
            "current_margin": reg_metrics(y_margin, cur_margin_preds),
            "tabfm_margin": reg_metrics(y_margin, tabfm_margin_preds),
            "current_total": reg_metrics(y_total, cur_total_preds),
            "tabfm_total": reg_metrics(y_total, tabfm_total_preds),
        }
        window_records.append(rec)

        all_game_ids.extend(game_ids.tolist())
        all_seasons.extend(seasons.tolist())
        all_y_win.extend(y_win.tolist())
        all_current_win.extend(cur_win_probs.tolist())
        all_tabfm_win.extend(test_proba.tolist())
        all_tabfm_platt.extend(tabfm_platt.tolist())
        all_tabfm_iso.extend(tabfm_iso.tolist())
        all_current_margin.extend(cur_margin_preds.tolist())
        all_tabfm_margin.extend(tabfm_margin_preds.tolist())
        all_y_margin.extend(y_margin.tolist())
        all_current_total.extend(cur_total_preds.tolist())
        all_tabfm_total.extend(tabfm_total_preds.tolist())
        all_y_total.extend(y_total.tolist())

        print(f"  current_win ll={rec['current_win']['log_loss']:.4f} tabfm ll={rec['tabfm_default_win']['log_loss']:.4f} "
              f"platt ll={rec['tabfm_platt_win']['log_loss']:.4f} iso ll={rec['tabfm_isotonic_win']['log_loss']:.4f}")
        print(f"  margin cur mae={rec['current_margin']['mae']:.4f} tabfm mae={rec['tabfm_margin']['mae']:.4f} "
              f"total cur mae={rec['current_total']['mae']:.4f} tabfm mae={rec['tabfm_total']['mae']:.4f}")
        print(f"  runtime={rec['runtime_sec']:.1f}s peak_rss={rec['peak_rss_mb']:.0f}MB")

    # pooled
    pooled = {
        "current_win": win_metrics(all_y_win, all_current_win),
        "tabfm_default_win": win_metrics(all_y_win, all_tabfm_win),
        "tabfm_platt_win": win_metrics(all_y_win, all_tabfm_platt),
        "tabfm_isotonic_win": win_metrics(all_y_win, all_tabfm_iso),
        "current_margin": pooled_reg(all_y_margin, all_current_margin),
        "tabfm_margin": pooled_reg(all_y_margin, all_tabfm_margin),
        "current_total": pooled_reg(all_y_total, all_current_total),
        "tabfm_total": pooled_reg(all_y_total, all_tabfm_total),
    }

    # bootstrap
    print("\n--- Bootstrap (pooled) ---")
    y_arr = np.asarray(all_y_win, dtype=float)
    cur_arr = np.asarray(all_current_win, dtype=float)
    tab_arr = np.asarray(all_tabfm_win, dtype=float)
    boot_pooled = paired_bootstrap(y_arr, cur_arr, tab_arr, "log_loss",
                                   n_resamples=N_RESAMPLES, seed=SEED)
    print(f"  pooled log_loss diff={boot_pooled['difference']:+.4f} 95%CI=[{boot_pooled['ci_low']:+.4f},{boot_pooled['ci_high']:+.4f}] improved={boot_pooled['improved']}")

    # per season
    seasons_sorted = sorted(set(all_seasons))
    boot_per_season = {}
    for s in seasons_sorted:
        mask = np.array([season == s for season in all_seasons])
        if mask.sum() < 20:
            continue
        ys = y_arr[mask]; cs = cur_arr[mask]; ts = tab_arr[mask]
        if len(np.unique(ys)) < 2:
            continue
        b = paired_bootstrap(ys, cs, ts, "log_loss", n_resamples=N_RESAMPLES, seed=SEED)
        boot_per_season[str(s)] = b
        print(f"  season {s}: n={int(mask.sum())} diff={b['difference']:+.4f} 95%CI=[{b['ci_low']:+.4f},{b['ci_high']:+.4f}] improved={b['improved']}")

    results = {
        "metadata": {
            "description": "NBA TabFM benchmark (measure only): zero-shot TabFM vs current NBA model, frozen history 2019-20 through 2024-25.",
            "seasons": "2019-20..2024-25",
            "n_games_after_feature_engineering": int(len(df)),
            "walkforward_windows": WINDOWS,
            "n_resamples": N_RESAMPLES,
            "seed": SEED,
            "n_estimators": N_EST,
            "production_sigma": production_sigma,
            "selected_context_policy": selected_policy,
        },
        "context_policy": {
            "scores_window0": policy_scores,
            "selected": selected_policy,
        },
        "home_win": {
            "per_window": window_records,
            "pooled": pooled,
        },
        "bootstrap_logloss_tabfm_minus_current": {
            "pooled": boot_pooled,
            "per_season": boot_per_season,
        },
        "runtime_summary": {
            "total_runtime_sec": float(sum(w["runtime_sec"] for w in window_records)),
            "peak_rss_mb": float(max(w["peak_rss_mb"] for w in window_records)),
        },
    }

    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    out_path = docs_dir / "nba_tabfm_benchmark_2019_2024.json"
    with open(out_path, "w") as f:
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
    print(f"\nResults written to {out_path}")

    # headline
    print("\n=== HEADLINE ===")
    print(f"selected policy: {selected_policy}")
    print(f"pooled current_win: ll={pooled['current_win']['log_loss']:.4f} brier={pooled['current_win']['brier']:.4f} auc={pooled['current_win']['auc']:.4f}")
    print(f"pooled tabfm_default: ll={pooled['tabfm_default_win']['log_loss']:.4f} brier={pooled['tabfm_default_win']['brier']:.4f} auc={pooled['tabfm_default_win']['auc']:.4f}")
    print(f"pooled tabfm_platt: ll={pooled['tabfm_platt_win']['log_loss']:.4f} brier={pooled['tabfm_platt_win']['brier']:.4f} auc={pooled['tabfm_platt_win']['auc']:.4f}")
    print(f"bootstrap pooled ll diff={boot_pooled['difference']:+.4f} 95%CI=[{boot_pooled['ci_low']:+.4f},{boot_pooled['ci_high']:+.4f}]")
    print(f"margin current mae={pooled['current_margin']['mae']:.4f} tabfm mae={pooled['tabfm_margin']['mae']:.4f}")
    print(f"total current mae={pooled['current_total']['mae']:.4f} tabfm mae={pooled['tabfm_total']['mae']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
