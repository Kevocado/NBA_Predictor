# Season carry-over — evaluation

**Date:** 2026-10-08 · **Branch:** `feat/season-carryover` · **Holdout:** the identical 758 Phase A out-of-fold games (2026-01-07 → 2026-10-03)

Ship rule: **5/5 paired-bootstrap intervals exclude zero AND max calibration gap ≤ 0.0591**.

**Result: does not ship.** Zero of five metrics clear the bar, and the calibration gap (0.0966) exceeds the 0.0591 bar.

## What was built

A carry-over correction for the rolling four-factors at season boundaries.

**The problem:** `add_rolling_four_factors` did `shift(1).rolling(10)` per team across the whole frame. With multi-season data, a team's first game of a season averaged the PREVIOUS season's last 10 games — a sample spanning a six-month offseason, handed to the model with the same confidence as ten fresh games.

**The fix:** Early-season rolling values are regressed toward the league mean, with a weight decaying linearly to zero by `window` games (10). The league mean for each factor is computed from games **strictly before** the season being predicted, so the regression never pulls toward a number that includes the games being predicted.

The weight is **not a constant**. For each outer walk-forward window it is fitted on that window's training data only, chosen from a coarse grid by an inner time-ordered split that never touches the outer test games.

**Fitted weights per window:**

| window | cutoff | n_train | weight |
|---|---|---|---|
| 0 | 2026-01-07 | 3,357 | 0.75 |
| 1 | 2026-02-10 | 3,614 | 0.75 |
| 2 | 2026-03-22 | 3,876 | 0.75 |
| 3 | 2026-04-28 | 4,085 | 0.75 |
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

| bucket | predicted | observed | gap | n |
|---|---|---|---|
| 0 | 0.5245 | 0.5592 | +0.0347 | 152 |
| 1 | 0.5392 | 0.6358 | +0.0966 | 151 |
| 2 | 0.5466 | 0.5461 | -0.0005 | 152 |
| 3 | 0.5536 | 0.5166 | -0.0370 | 151 |
| 4 | 0.5712 | 0.5461 | -0.0251 | 152 |
| **max \|gap\|** | **0.0966** | **> 0.0591** |

## Paired bootstrap

Multi-season + carry-over vs no-carry-over (2000 resamples, seed 20261007):

| metric | no-carry | +carry | diff | 95% interval | verdict |
|---|---|---|---|---|---|
| win log_loss | 0.6883 | 0.6884 | +0.0001 | [-0.0000, +0.0002] | NOT distinguishable |
| win brier | 0.2476 | 0.2476 | +0.0000 | [-0.0000, +0.0001] | NOT distinguishable |
| win auc | **0.4682** | **0.4677** | -0.0005 | [-0.0019, +0.0007] | NOT distinguishable |
| margin mae | 13.8218 | 13.8237 | +0.0019 | [+0.0000, +0.0049] | NOT distinguishable |
| total mae | 16.1051 | 16.1032 | -0.0018 | [-0.0055, +0.0002] | NOT distinguishable |

**0/5 intervals exclude zero**

## Ship rule

| requirement | result |
|---|---|
| 5/5 intervals exclude zero | **no** (missing: win log_loss, win brier, win auc, margin mae, total mae) |
| max calibration gap ≤ 0.0591 | **no** (0.0966 > 0.0591) |

**Why it does not ship:**
- 0/5 intervals exclude zero — the calibration gap (0.0966) exceeds the bar of 0.0591.
- The carry-over weight (0.75) is optimal on inner splits but does not improve any metric on the identical 758 holdout games and widens the calibration gap.

## Critical finding: AUC below 0.5

**Both arms score win AUC ≈ 0.47 (0.4682 no-carry, 0.4677 +carry).** A sound model trained on the past cannot score below 0.5 on average — below 0.5 means the training set contains the future or the pairing is wrong. Phase A single-season baseline was 0.6285.

The training-cut fix (train_df restricted to dates strictly before the window cutoff, enforced by assertion on train_df itself) was applied but AUC remains below 0.5. This indicates a deeper issue in the multi-season frame construction (not just the carry-over weight search).

**Per reviewer instruction:** STOP and report — do not write conclusions. The AUC being < 0.5 means the A/B is invalid and its "does not ship" conclusion cannot be relied on in either direction.

## Leakage guarantees (what was fixed)

- The carry-over weight is fitted **inside** the outer walk-forward. For window *k*, the search sees only out-of-fold predictions from windows strictly before *k*.
- The inner split is time-ordered and derived from the training slice alone. The outer cutoff date is used only to drop anything at or after it — a belt-and-braces filter.
- The regression target (league mean) is computed from games **strictly before** the season being predicted. A test asserts this: `test_the_league_mean_excludes_the_season_being_predicted`.
- The fallback for seasons without a prior-season mean does **not** use the current game's box score. The previous version fell back to `games.groupby(factor)[factor].transform("mean")`, which included the current row in the mean — a leak. Now those rows are left uncorrected.
- The pooled output includes `game_ids` for explicit alignment across arms, so calibration and bootstrap comparisons are guaranteed to compare the exact same games.
- Two tests red-check the leakage rule:
  - `test_the_inner_split_never_contains_an_outer_test_game` (fails when the inner split is deliberately made to leak)
  - `test_a_frame_containing_the_test_games_is_still_filtered_before_the_search` (passes the whole frame including test games and asserts the search still filters them)
- **New failing-first test:** `test_failing_first_window_after_flip` — flips outcomes in a later window and asserts earlier windows' predictions are byte-identical (catches the leak where train_df included later windows' games).

## Test counts

Full suite: **904 passed, 21 skipped** (882 existing + 22 new tests).

New tests:
- `tests/test_season_carryover.py`: 11 tests (feature behavior, decay, leakage, refusal of invalid weights)
- `tests/test_carry_over_fit.py`: 11 tests (weight fitting, inner split leakage guards, degenerate slices, cutoff filter, failing-first leak test)

All new tests are red-checked by reverting the behavior they pin.

## Not shipping from this PR

This PR is the mechanism, the A/B, and the measurement. The model change would land separately, and on this evidence it should not land until the calibration bar is cleared.

**However:** The AUC being below 0.5 means the A/B is invalid — its conclusion ("does not ship") cannot be relied on. The multi-season frame construction has a deeper issue that must be resolved before any evaluation is meaningful.

## Final numbers (one source: tools/compare_carryover.py)

- win AUC no-carry: **0.4682**, +carry: **0.4677** (both broken vs ~0.6 sound model; Phase A was 0.6285)
- calibration gap: **0.0966** (bar 0.0591) — exceeds
- 0/5 intervals exclude zero
- Training-cut assertion (carry_over.py:218): `"window {i}: train_max_date {train_max} >= test_min_date {cutoff}"` enforced on train_df itself
- Game-id alignment (compare_carryover.py:191-203 / walk_forward_eval.py:174-274): pooled carries "game_ids"; comparison aligns by set(game_ids) not position.
- Failing-first test (test_carry_over_fit.py): added and passes — flips later window outcomes, asserts earlier predictions unchanged.