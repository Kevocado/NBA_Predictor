"""Do the lagged-availability columns earn their place? (confirmation run, repo code path)

Baseline = the repo's FEATURE_COLUMNS minus `AVAILABILITY_COLUMNS`; candidate = FEATURE_COLUMNS. Both frames come from
the real `build_training_frame(..., players=...)`, scored with the real `default_candidates` under the real
`ship_decision` rule (nothing here changes the rule or its gates). Walk-forward over 2018-19..2024-25; only the
out-of-fold games from 2022-23 on are scored (the seasons the feature was NOT chosen on).

A tool, not a test: it reads the gitignored `data/cache/training/games_all.json` and the ESPN player box-score cache.

Run:
    uv run --python 3.13 python tools/compare_availability.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_predictor import config  # noqa: E402
from nba_predictor.features.availability import AVAILABILITY_COLUMNS  # noqa: E402
from nba_predictor.features.build import build_training_frame  # noqa: E402
from nba_predictor.models.artifact_gate import calibration_gap  # noqa: E402
from nba_predictor.models.candidate_race import default_candidates  # noqa: E402
from nba_predictor.models.evaluate.paired_bootstrap import paired_bootstrap, paired_gap_bootstrap  # noqa: E402
from nba_predictor.models.evaluate.ship_rule import ship_decision  # noqa: E402
from nba_predictor.models.evaluate.walk_forward_eval import (  # noqa: E402
    expanding_windows,
    walk_forward_metrics,
    walk_forward_regression,
)
from nba_predictor.pipeline.ingest import to_player_training_frame  # noqa: E402

GAMES = Path("data/cache/training/games_all.json")
END, CUT, WINDOWS = "2025-08-01", "2022-08-01", 5
N_RES, SEED = 1500, 20261015


def _players(games_ids: set[str]) -> pd.DataFrame:
    schedule = []
    for f in sorted(config.CACHE_DIR.glob("espn/scoreboard_*.json")):
        schedule += json.loads(f.read_text())["games"]
    schedule = [g for g in schedule if g.get("completed") and g["game_id"] in games_ids]
    pb = {}
    for g in schedule:
        f = config.CACHE_DIR / "espn" / f"player_boxscore_{g['game_id']}.json"
        if f.exists():
            pb[g["game_id"]] = json.loads(f.read_text())["rows"]
    return to_player_training_frame(schedule, pb)


def _arm(F, cols):
    c = default_candidates(cols)
    return {
        "win": walk_forward_metrics(F, model_factory=lambda tr: c["logistic"]["win"](tr), windows=WINDOWS)["pooled"],
        "margin": walk_forward_regression(F, model_factory=lambda tr: c["ridge"]["margin"](tr), target="home_margin",
                                          windows=WINDOWS)["pooled"],
        "total": walk_forward_regression(F, model_factory=lambda tr: c["ridge"]["total"](tr), target="home_total",
                                         windows=WINDOWS, fixed_baseline=float(F.home_total.mean()))["pooled"],
    }


def main() -> int:
    if not GAMES.exists():
        print(f"no games cache at {GAMES}; this is a local measurement, not a test")
        return 1
    raw = pd.DataFrame(json.loads(GAMES.read_text()))
    raw = raw[raw.game_date < END].reset_index(drop=True)
    players = _players(set(raw.game_id))
    F, cols = build_training_frame(raw, players=players)
    F = F.assign(home_margin=F.home_pts - F.away_pts, home_total=F.home_pts + F.away_pts)
    F = F.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    base_cols = [c for c in cols if c not in AVAILABILITY_COLUMNS]
    base, cand = _arm(F, base_cols), _arm(F, cols)

    pos = np.concatenate([te for _, te in expanding_windows(F["game_date"], WINDOWS)])
    dates = F["game_date"].to_numpy()[pos]
    mask = dates >= CUT
    y = np.asarray(base["win"]["y"])[mask]
    g = lambda d, k: np.asarray(d[k]["preds"])[mask]  # noqa: E731
    boot = {
        "win log_loss": paired_bootstrap(y, g(base, "win"), g(cand, "win"), "log_loss", n_resamples=N_RES, seed=SEED),
        "win brier": paired_bootstrap(y, g(base, "win"), g(cand, "win"), "brier", n_resamples=N_RES, seed=SEED),
        "win auc": paired_bootstrap(y, g(base, "win"), g(cand, "win"), "auc", n_resamples=N_RES, seed=SEED),
        "margin mae": paired_bootstrap(np.asarray(base["margin"]["y"])[mask], g(base, "margin"), g(cand, "margin"),
                                       "mae", n_resamples=N_RES, seed=SEED),
        "total mae": paired_bootstrap(np.asarray(base["total"]["y"])[mask], g(base, "total"), g(cand, "total"),
                                      "mae", n_resamples=N_RES, seed=SEED),
    }
    gb, gc = calibration_gap(y, g(base, "win"), 5), calibration_gap(y, g(cand, "win"), 5)
    widened = paired_gap_bootstrap(y, g(base, "win"), g(cand, "win"), n_resamples=500, seed=SEED)["widened"]
    dec = ship_decision(boot, gap=gc, baseline_gap=gb, gap_widened=widened)
    print(f"n={int(mask.sum())} held-out games ({dates[mask].min()} .. {dates[mask].max()}), "
          f"{len(base_cols)} -> {len(cols)} features")
    for k, o in boot.items():
        print(f"  {k:<13}{o['baseline']:>9.4f}{o['candidate']:>9.4f}{o['difference']:>+9.4f}  "
              f"[{o['ci_low']:+.4f},{o['ci_high']:+.4f}] {'IMPROVED' if o['improved'] else 'n.s.'}")
    print(f"  cal gap {gb:.4f} -> {gc:.4f} (real widening: {widened}); improved={dec['improved']} "
          f"worse={dec['worse']} auc_ok={dec['auc_ok']} => {'SHIPS' if dec['ships'] else 'does NOT ship'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
