# Scratch: Phase A real-data measurements

Feeds `docs/nba-parity-evaluation-2026-10.md` (Task 11). Not a report — numbers
collected as tasks land.

## Task 3 — real XGBoost win model, expanding-window walk-forward

Data: `data/cache/training/games.json`, 1,390 games -> 1,368 usable rows,
24 features from `features/build.py`. 5 date-aligned windows (the evaluator
cuts on dates; 1,368 games across ~190 dates).

| window | n_train | train_max | n_test | test_min | log_loss | auc |
|---|---|---|---|---|---|---|
| 0 | 610 | 2026-01-09 | 250 | 2026-01-10 | 0.7569 | 0.513 |
| 1 | 860 | 2026-02-11 | 249 | 2026-02-12 | 0.7263 | 0.586 |
| 2 | 1109 | 2026-03-22 | 204 | 2026-03-23 | 0.6419 | 0.666 |
| 3 | 1313 | 2026-04-27 | 51 | 2026-04-28 | 0.6180 | 0.736 |
| 4 | 1364 | 2026-06-10 | 4 | 2026-06-13 | 0.6283 | 0.667 |

**Pooled: log_loss 0.7059 | naive (home base) 0.6864 | coin-flip 0.6931**
brier 0.2516 | auc 0.5912 | accuracy 0.5844 (n=758)

### Reads

1. **It does not beat naive.** Worse than the home-win base rate (0.6864) and
   worse than a coin flip (0.6931). AUC 0.5912 is near chance.
2. **The old 0.6354 accuracy was flattering.** The manifest reported accuracy
   from one 80/20 `chronological_split`. Walk-forward accuracy on out-of-fold
   data is 0.5844. Same model, honest method, ~5 points lower.
3. **It improves with more training data, though not monotonically.**
   log_loss 0.7569 -> 0.6180 as training goes 610 -> 1313 games, then back up to
   0.6283 in the final window — which is 4 games, so that last number is noise
   rather than a reversal. The trend through the meaningful windows is
   downward, and the early-season weakness is a sample-size problem, which is
   what a candidate race (Task 5) may or may not fix — Ridge with 24 features on
   600 rows may beat a 200-tree depth-4 booster outright. NFL's Ridge beat
   XGBoost for exactly this reason.
4. **Window 4 is 4 games.** Date-aligned slicing leaves a sliver at the end;
   it contributes 4 rows to the pooled figure. Not material at n=758, but the
   report should not quote a per-window table as if every row were equal.

### Task 4 — margin/total walk-forward (via `run_retrain_pipeline`, 1,390 real games)

Note: this is scored on `train_games` (the 80% chronological split), so it is not directly
comparable to the Task 3 table above, which used the full frame. Task 11 must quote one
consistently.

| model | metric | walk-forward | naive baseline | verdict |
|---|---|---|---|---|
| win | log-loss | 0.7712 | coinflip 0.6931 / train-base | loses |
| win | brier | 0.2753 | — | — |
| win | auc | 0.5496 | 0.5 | ~chance |
| margin | MAE | **13.461** | **12.861** (home −3 fixed) | **loses** |
| total | MAE | **16.904** | — | — |

Holdout (old path, retained for reference): accuracy 0.6583, margin MAE 13.698, total MAE 18.540.

### Reads

1. **The margin model does not beat "home wins by 3".** Walk-forward MAE 13.461 against a
   fixed −3 baseline of 12.861. The old spec text called margin MAE 13.9 "at-or-above naive
   scale (margin σ≈12)"; walk-forward puts it concretely: the model is *worse than a constant
   −3*. That is the honest headline for Phase A.
2. **Total improves on the holdout figure** (16.904 vs 18.540) — different estimator, not a
   regression, but not comparable either.
3. The win AUC of 0.5496 here is even weaker than Task 3's 0.5912 on the full frame, and the
   log-loss worse (0.7712 vs 0.7059). Consistent with the sample-size story: this run trains on
   80% of the data, and each window refits from scratch on a growing slice.

This is what makes Task 5 (candidate race) the load-bearing next step. If Ridge beats a
200-tree depth-4 booster on 24 features at these sample sizes, that is the whole ballgame —
and it is exactly what happened in the NFL repo.

Not yet measured: race winner (Task 5), residual-σ calibration (Task 6), prop holdout (8-9).