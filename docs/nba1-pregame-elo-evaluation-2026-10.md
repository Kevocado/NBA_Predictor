# NBA-1: pre-game Elo features: evaluation

**Result: does not ship.** The paired-bootstrap interval on the per-game log-loss difference crosses zero, and ECE is worse. The feature is not wired into any served frame. Raw output: `docs/nba1_pregame_elo_2019_2024.json`. Tool: `tools/nba1_pregame_elo.py`.

## What was tested

- **Games and windows:** identical to NBA-0 (`tools/nba0_measure.py`): 2019-20 through 2024-25 regular season, 4 expanding windows, 4,564 pooled out-of-fold games.
- **Served probability:** `norm.cdf(predicted_margin / sigma)`, from a StandardScaler + Ridge margin model. Sigma is the window-0 training MAE times `MAE_TO_SIGMA`, held constant for all windows, as in NBA-0.
- **Baseline:** the production feature set (`FEATURE_COLUMNS`, 28 columns).
- **Candidate:** baseline plus `pregame_elo_diff_s`, each team's rating as of tip-off, divided by 100.
- **Rating rule:** a game's ratings are read before any result on its date is folded in. Teams regress 1/3 of the way to 1500 at their first game of a new season.
- **K and home-court advantage:** chosen per window from K in {10, 20, 30} and HCA in {0, 50, 100}. The choice uses only that window's training data, scored on its last season. Selected values were K=20/HCA=50, K=10/HCA=50, K=10/HCA=50, K=20/HCA=0.

## Result

| metric | baseline | + pre-game Elo | diff (candidate - baseline) | 95% interval | verdict |
|---|---|---|---|---|---|
| log-loss | 0.62255 | 0.62153 | -0.00102 | [-0.00236, +0.00030] | crosses zero: fails |
| Brier | 0.21665 | 0.21618 | -0.00046 | [-0.00104, +0.00009] | crosses zero |
| AUC | 0.70243 | 0.70356 | +0.00113 | [-0.00082, +0.00311] | crosses zero; point estimate not worse |
| ECE (10 bins) | 0.01085 | 0.01302 | +0.00217 | not bootstrapped | worse |
| calibration gap (5 buckets) | 0.02173 | 0.01804 | -0.00369 | not bootstrapped | not worse |

Bootstrap: 2,000 paired resamples, seed 20261007, all resamples used.

Runtime: 53 s for the full walk-forward and bootstrap on the 2019-2024 history.

Reference: NBA-0's served probability, with the unscaled production ridge, scored log-loss 0.6224. The baseline arm here scores 0.6226, so the harness reproduces it.

## Why it does not ship

The gate needs the log-loss interval entirely below zero. It is centred slightly below zero but its upper end is above zero, so there is no evidence of improvement at this sample size. ECE also gets worse. AUC and the 5-bucket calibration gap move the right way, but the rule requires all of them together.

## Note on the existing feature

`features/team_features.py` already computes a causal Elo, exposed as `elo_diff_s` in `FEATURE_COLUMNS`. It uses K=20, no home-court term, no season regression, and starts at 1500. Each row records the rating before its own result is folded in. The new module is therefore a variant of an existing feature, not a first Elo. The result says the variant does not add measurable value on this window set.

## Leak control

- `tests/test_pregame_elo.py` checks that flipping a game's own result, flipping a later game's result, or flipping another game on the same date leaves that game's pre-game ratings unchanged. It also checks that a result does reach later games.
- A mutant that folds a game's own result in before reading it fails this test.

## Status

Docs-only. `src/nba_predictor/features/pregame_elo.py` and `tools/nba1_pregame_elo.py` are reusable, but no frame, model or served path reads them.
