"""Task 4(d): does averaging Ridge and XGBoost margins beat picking a winner?

Audit followup `2026-10-06-nba-audit-followup.md` Task 4(d). The candidate race is
winner-take-all -- `_pick_winner` takes the single best per target by the deciding
metric. This asks whether the plain mean of the two margin models does better than
either alone, on the identical held-out games, with the paired bootstrap the ship
rule requires.

The average is built the way a real candidate would be: both regressors fitted
inside the walk-forward on the training slice only, then averaged, so nothing here
sees a test game.

This is a **tool**, not a test, and deliberately so. It reads
`data/cache/training/games.json`, which is gitignored -- the box scores are not
committed. A test that reads it would pass locally and error in CI, which is how a
green run stops meaning anything. `tools/compare_carryover.py` is the same shape
for the same reason.

Run:
    uv run --python 3.13 python tools/compare_margin_average.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_predictor.features.build import build_training_frame  # noqa: E402
from nba_predictor.models.candidate_race import (  # noqa: E402
    default_candidates,
    make_ridge_regressor_factory,
    make_xgb_regressor_factory,
)
from nba_predictor.models.evaluate.paired_bootstrap import paired_bootstrap  # noqa: E402
from nba_predictor.models.evaluate.walk_forward_eval import walk_forward_regression  # noqa: E402

TARGET = "home_margin"
N_RESAMPLES = 2000
SEED = 20261009
CACHE = Path("data/cache/training/games.json")


def _averaged_factory(feature_cols, target):
    """Mean of the Ridge and XGBoost regressors for `target`.

    Both are fitted on the training slice handed to the factory, exactly as the
    race fits them, so neither sees a test game.
    """
    ridge = make_ridge_regressor_factory(feature_cols, target)
    xgb = make_xgb_regressor_factory(feature_cols, target)

    def factory(train_df):
        r = ridge(train_df)
        x = xgb(train_df)

        def predict(test_df):
            return 0.5 * (np.asarray(r(test_df)) + np.asarray(x(test_df)))

        return predict

    return factory


def main() -> int:
    if not CACHE.exists():
        print(f"no training cache at {CACHE}; this is a local measurement, not a test")
        return 1

    raw = __import__("pandas").DataFrame(json.loads(CACHE.read_text()))
    frame, cols = build_training_frame(raw)
    frame = frame.assign(
        home_margin=frame.home_pts - frame.away_pts,
        home_total=frame.home_pts + frame.away_pts,
    )

    candidates = default_candidates(cols)
    results = {}
    for name, factory in (
        ("Ridge", candidates["ridge"]["margin"]),
        ("XGBoost", candidates["xgboost"]["margin"]),
        ("average", _averaged_factory(cols, TARGET)),
    ):
        results[name] = walk_forward_regression(
            frame, model_factory=factory, target=TARGET, windows=4)

    print(f"\nframe: {len(frame)} games, pooled n={results['Ridge']['pooled']['n']}")
    print(f"\n=== Task 4(d): average Ridge and XGBoost margins")
    print(f"  {'arm':<12}{'margin MAE':>12}")
    for name in ("Ridge", "XGBoost", "average"):
        print(f"  {name:<12}{results[name]['pooled']['mae']:>12.4f}")

    best = min(("Ridge", "XGBoost"), key=lambda n: results[n]["pooled"]["mae"])
    base, avg = results[best], results["average"]
    delta = avg["pooled"]["mae"] - base["pooled"]["mae"]
    print(f"\n  average minus the best single ({best}): {delta:+.4f}")

    out = paired_bootstrap(
        np.asarray(base["pooled"]["y"]),
        np.asarray(base["pooled"]["preds"]),
        np.asarray(avg["pooled"]["preds"]),
        "mae", n_resamples=N_RESAMPLES, seed=SEED,
    )
    ci = f"[{out['ci_low']:+.4f}, {out['ci_high']:+.4f}]"
    print(f"\n=== paired bootstrap: average vs {best}, {N_RESAMPLES} resamples, seed {SEED}")
    print(f"  {best:<12}{out['baseline']:>10.4f}")
    print(f"  {'average':<12}{out['candidate']:>10.4f}")
    print(f"  diff {out['difference']:+.4f}   95% interval {ci}")
    print(f"  verdict: {'IMPROVED' if out['improved'] else 'NOT distinguishable'}")

    ships = out["improved"]
    print(f"\n=== ship rule: the candidate must BEAT the current winner")
    print(f"  Task 4(d)    {'SHIPS' if ships else 'does NOT ship'}")
    if not ships:
        print(f"      - the interval crosses zero; a 758-game holdout cannot resolve "
              f"{abs(out['difference']):.4f} of margin MAE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
