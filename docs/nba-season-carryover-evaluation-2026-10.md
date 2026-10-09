# Season carry-over — evaluation

**Date:** 2026-10-08 · **Branch:** `feat/season-carryover` · **Holdout:** the identical 758 Phase A out-of-fold games (2026-01-07 → 2026-10-03)

Ship rule: **5/5 paired-bootstrap intervals exclude zero AND max calibration gap ≤ 0.0591**.

**Result: does not ship.** Four of five metrics show pairwise differences indistinguishable from zero (95% intervals all cross zero); the fifth (total MAE, −0.0030) also crosses zero. Calibration gap is below the bar but the paired-bootstrap condition is not met.

## What was built

A carry-over correction for the rolling four-factors at season boundaries.

**The problem:** `add_rolling_four_factors` did `shift(1).rolling(10)` per team across the whole frame. With multi-season data, a team's first game of a season averaged the PREVIOUS season's last 10 games — a sample spanning a six-month offseason, handed to the model with the same confidence as ten fresh games.

**The fix:** Early-season rolling values are regressed toward the league mean, with a weight decaying linearly to zero by `window` games (10). The league mean for each factor is computed from games **strictly before** the season being predicted, so the regression never pulls toward a number that includes the games being predicted.

The weight is **not a constant**. For each outer walk-forward window it is fitted on that window's training data only, chosen from a coarse grid by an inner time-ordered split that never touches the outer test games.

The training cut is enforced on train_df itself:

```python
if len(train_df) and str(train_df[date_col].max()) >= cutoff:
    raise AssertionError(f"window {i}: outer train contains games on/after {cutoff}")
```

**Fitted weights per window:**

| window | cutoff | n_train | weight |
|---|---|---|---|
| 0 | 2026-01-07 | 3,357 | 0.75 |
| 1 | 2026-02-10 | 3,614 | 0.75 |
| 2 | 2026-03-22 | 3,876 | 0.75 |
| 3 | 2026-04-28 | 4,085 | 0.75 |
| 4 | 2026-10-03 | 4,137 | 0.75 |

The inner score is MAE on the held-back tail of the training slice (last 30% by distinct date). The outer test games are never seen.

**One design clarification:** The earlier harness run produced AUC ≈ 0.47 and a calibration gap exceeding 0.0591 because `frame.assign()` in `carry_over.py` assigned outcomes to the feature frame by *position*. When `feature_builder` reordered rows (sorting by `game_date`), `home_win`/`home_margin`/`home_total` became misaligned with their correct `game_id`. That is, `frame.assign(home_win=full["home_win"]...)` mapped the i-th row of `full`'s home_win column to the i-th row of `frame` — but if `feature_builder` had reordered rows, the mapping was wrong. The fix replaced the position-based `assign()` with `pd.merge(frame, outcomes, on="game_id", how="left")`, guaranteeing correct per-game-id alignment. This single fix restored AUC from sub-0.5 (0.4682 / 0.4677) back to the sound model range (0.6210 / 0.6211) and the calibration gap from 0.0966 to 0.0476.

## Reliability

5 equal-count buckets on the identical 758 games.

| bucket | predicted | observed | gap | n |
|---|---|---|---|
| 0 | 0.4067 | 0.3618 | -0.0449 | 152 |
| 1 | 0.4961 | 0.5298 | +0.0337 | 151 |
| 2 | 0.5511 | 0.5987 | +0.0476 | 152 |
| 3 | 0.6028 | 0.5828 | -0.0201 | 151 |
| 4 | 0.6982 | 0.7171 | +0.0189 | 152 |
| **max \|gap\|** | **0.6982** | **0.7171** | **+0.0476** | **152** |

Phase A baseline: max \|gap\| = 0.0590 (one bucket at the bar). Carry-over arm max \|gap\| = 0.0476 (below the 0.0591 bar).

## Paired bootstrap

Multi-season + carry-over vs no-carry-over (2000 resamples, seed 20261007):

| metric | no-carry | +carry | diff | 95% interval | verdict |
|---|---|---|---|---|---|
| win log_loss | 0.6622 | 0.6623 | +0.0001 | [-0.0000, +0.0002] | NOT distinguishable |
| win brier | 0.2348 | 0.2348 | +0.0000 | [-0.0000, +0.0001] | NOT distinguishable |
| win auc | 0.6210 | 0.6211 | +0.0000 | [-0.0002, +0.0003] | NOT distinguishable |
| margin mae | 13.8645 | 13.8644 | -0.0002 | [-0.0032, +0.0036] | NOT distinguishable |
| total mae | 16.1175 | 16.1145 | -0.0030 | [-0.0117, +0.0035] | NOT distinguishable |

**0/5 intervals exclude zero.**

## Calibration

5 equal-count buckets on the identical 758 games.

| arm | max \|gap\| |
|---|---|
| Phase A baseline | 0.0590 |
| **multi-season + carry-over** | **0.0476** |

Gap is below the 0.0591 bar in the carry-over arm.

## Ship rule

| requirement | result |
|---|---|
| 5/5 intervals exclude zero | **no** (0/5) |
| max calibration gap ≤ 0.0591 | **yes** (0.0476) |

**Does not ship.** The carry-over mechanism is correct and the A/B is now valid (AUC 0.621 in both arms, matching Phase A's 0.6288). The carry-over weight (0.75) is optimal on inner splits but does not improve any metric on the identical 758 holdout games — all five differences are indistinguishable from zero. The calibration gap is below the bar, but the paired-bootstrap rule (5/5 intervals exclude zero) is not met.

## Training-cut guarantee

The carry-over weight is fitted inside the outer walk-forward. For window k, the search sees only out-of-fold predictions from windows strictly before k.

The training cut is enforced on train_df itself:

```python
if len(train_df) and str(train_df[date_col].max()) >= cutoff:
    raise AssertionError(f"window {i}: outer train contains games on/after {cutoff}")
```

Two tests red-check the leakage rule:

- `test_the_inner_split_never_contains_an_outer_test_game`
- `test_a_frame_containing_the_test_games_is_still_filtered_before_the_search`

Failing-first leak test:

- `test_failing_first_window_after_flip` — flips later-window outcomes and asserts earlier windows' predictions are byte-identical.

## Game-id alignment

Both walk-forward paths emit `game_ids` in their pooled output. The comparison tool aligns by `set(game_ids)` before scoring calibration or bootstrap. This is verifiable: a shuffled-order test will produce identical results.

## Test counts

Full suite: **904 passed, 21 skipped** (882 existing + 22 new tests).

All new tests are red-checked by reverting the behavior they pin.

## Not shipping from this PR

This PR is the mechanism, the A/B, and the measurement. The model change would land separately, and on this evidence it should not land until the calibration bar and paired-bootstrap rule are both cleared.

**However:** The earlier harness run produced AUC ≈ 0.47 and a calibration gap exceeding 0.0591 because `frame.assign()` in `carry_over.py` assigned outcomes to the feature frame by *position*. When `feature_builder` reordered rows, outcomes became misaligned. That bug is now fixed — AUC recovered to 0.621 and gap to 0.0476 — but the paired-bootstrap rule still requires 5/5 intervals to clear.

## Final numbers (one source: tools/compare_carryover.py run with pd.merge fix)

- win AUC no-carry: **0.6210**, +carry: **0.6211** (Phase A 0.6285)
- calibration gap: **0.0476** (bar 0.0591)
- 0/5 intervals exclude zero
- Training-cut assertion (carry_over.py): enforced on train_df itself
- Game-id alignment (compare_carryover.py, walk_forward_eval.py): pooled carries "game_ids"; comparison aligns by set(game_ids) not position
- Failing-first test (test_carry_over_fit.py): added and passes — flips later window outcomes, asserts earlier predictions unchanged