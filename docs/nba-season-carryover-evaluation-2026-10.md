# Season carry-over — evaluation

**Date:** 2026-10-08 · **Branch:** `feat/season-carryover` · **Holdout:** the identical 758 Phase A out-of-fold games (2026-01-07 → 2026-10-03)

Ship rule: **5/5 paired-bootstrap intervals exclude zero AND max calibration gap ≤ 0.0591**.

**Result: does not ship.** Zero of five metrics clear the bar, and the calibration gap (0.0732) exceeds the 0.0591 bar.

## What was built

A carry-over correction for the rolling four-factors at season boundaries.

**The problem:** `add_rolling_four_factors` did `shift(1).rolling(10)` per team across the whole frame. With multi-season data, a team's first game of a season averaged the PREVIOUS season's last 10 games — a sample spanning a six-month offseason, handed to the model with the same confidence as ten fresh games.

**The fix:** Early-season rolling values are regressed toward the league mean, with a weight decaying linearly to zero by `window` games (10). The league mean for each factor is computed from games **strictly before** the season being predicted, so the regression never pulls toward a number that includes the games being predicted.

The weight is **not a constant**. For each outer walk-forward window it is fitted on that window's training data only, chosen from a coarse grid by an inner time-ordered split that never touches the outer test games.

**Fitted weights per window:**

| window | cutoff | n_train | weight |
|---|---|---|---|
| 0 | 2026-01-07 | 3,883 | 0.75 |
| 1 | 2026-02-10 | 3,878 | 0.75 |
| 2 | 2026-03-22 | 3,931 | 0.75 |
| 3 | 2026-04-28 | 4,088 | 0.75 |
| 4 | 2026-10-03 | 4,137 | 0.75 |

**Inner MAE by weight (window 0 example):**

| weight | inner MAE |
|---|---|
| 0.0 | 12.718 |
| 0.15 | 12.715 |
| 0.3 | 12.711 |
| 0.45 | 12.707 |
| 0.6 | 12.704 |
| **0.75** | **12.701** ← chosen |

The inner score is MAE on the held-back tail of the training slice (last 30% by distinct date). The outer test games are never seen.

## Reliability

5 equal-count buckets on the identical 758 games.

| bucket | n | baseline (Phase A) | multi-season (no carry-over) | **+carry-over** |
|---|---|---|---|---|
| 0 | 152 | -0.0394 | +0.0717 | +0.0721 |
| 1 | 151 | -0.0156 | +0.0730 | +0.0732 |
| 2 | 152 | +0.0590 | +0.0292 | +0.0294 |
| 3 | 151 | +0.0291 | -0.0527 | -0.0525 |
| 4 | 152 | -0.0153 | -0.0445 | -0.0442 |
| **max \|gap\|** | | **0.0590** | **0.0730** | **0.0732** |

The calibration gap **widens slightly** (0.0730 → 0.0732) and remains above the 0.0591 bar.

## Paired bootstrap

2,000 resamples, seed 20261007, one draw of game indices applied to both arms on the identical 758 games.

**+carry-over vs no-carry-over**

| metric | no-carry | +carry | diff | 95% interval | verdict |
|---|---|---|---|---|---|
| win log-loss | 0.6880 | 0.6881 | +0.0001 | [-0.0000, +0.0002] | NOT distinguishable |
| win Brier | 0.2474 | 0.2475 | +0.0000 | [-0.0000, +0.0001] | NOT distinguishable |
| win AUC | 0.4607 | 0.4605 | -0.0002 | [-0.0014, +0.0010] | NOT distinguishable |
| margin MAE | 13.8233 | 13.8250 | +0.0017 | [-0.0002, +0.0047] | NOT distinguishable |
| total MAE | 16.1111 | 16.1092 | -0.0019 | [-0.0055, +0.0000] | NOT distinguishable |

**0/5 intervals exclude zero. The calibration gap (0.0732) exceeds 0.0591.**

## Ship rule

| requirement | result |
|---|---|
| 5/5 intervals exclude zero | **no** (0/5) |
| max calibration gap ≤ 0.0591 | **no** (0.0732) |

**Does not ship.** The data says a strong carry-over (weight 0.75) is optimal on the inner splits, but on the identical 758 holdout games it does not improve any metric and the calibration gap is unchanged (actually marginally worse).

## Leakage guarantees

- The carry-over weight is fitted **inside** the outer walk-forward. For window *k*, the search sees only out-of-fold predictions from windows strictly before *k*.
- The inner split is time-ordered and derived from the training slice alone. The outer cutoff date is used only to drop anything at or after it — a belt-and-braces filter.
- The regression target (league mean) is computed from games strictly **before** the season being predicted. A test asserts this: `test_the_league_mean_excludes_the_season_being_predicted`.
- The fallback for seasons without a prior-season mean does NOT use the current game's box score. The previous version fell back to `games.groupby(factor)[factor].transform("mean")`, which included the current row in the mean — a leak. Now those rows are left uncorrected.
- Two tests red-check the leakage rule:
  - `test_the_inner_split_never_contains_an_outer_test_game` (fails when the inner split is deliberately made to leak)
  - `test_a_frame_containing_the_test_games_is_still_filtered_before_the_search` (passes the whole frame including test games and asserts the search still filters them)

## Test counts

Full suite: **903 passed, 21 skipped** (882 existing + 21 new tests).

New tests:
- `tests/test_season_carryover.py`: 11 tests (feature behavior, decay, leakage, refusal of invalid weights)
- `tests/test_carry_over_fit.py`: 10 tests (weight fitting, inner split leakage guards, degenerate slices, cutoff filter)

All new tests are red-checked by reverting the behavior they pin.

Full suite: **903 passed, 21 skipped**.

## Not shipping from this PR

This PR is the mechanism, the A/B, and the measurement. The model change would land separately, and on this evidence it should not land until the calibration bar is cleared.

The calibration gap is a known, reproducible defect in bucket 0 (the model under-predicts home wins in the lowest-probability bucket). A calibration step (Platt or isotonic) fitted on out-of-fold predictions is the standard remedy and should be evaluated on the same 758 before anything else is tried.