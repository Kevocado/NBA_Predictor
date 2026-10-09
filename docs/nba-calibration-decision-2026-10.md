# Calibration decision — the AUC bar

**Date:** 2026-10-09 · **Branch:** `feat/calibration-decision` · **Decides:** the AUC bar in the ship rule, and where Platt stands against it.

This is a decision, not a measurement. Every number it rests on is either reproduced here or quoted from `nba-calibration-evaluation-2026-10.md`, which is where the measurement lives.

**The decision, stated first: the AUC bar requires the 95% interval to exclude a *decline*. It no longer requires it to exclude zero. Platt still does not ship, and is not relaxed into shipping.**

## Why the bar was wrong as written

The ship rule said "all five paired-bootstrap intervals exclude zero", and AUC is one of the five. For a calibrator that asks the instrument to do something it cannot do.

A probability recalibration is a monotone re-expression of the same ranking. It changes *what the numbers mean*, not *which games the model prefers*. So gating it on a ranking metric is gating it on a quantity it leaves alone by construction.

The doc claimed both "Platt is a strictly monotone map, so it is AUC-preserving to 1e-12" and a pooled AUC move of 0.6288 → 0.6447. Those two sit together only because the calibrator is fitted *per window* inside the walk-forward. `tests/test_platt_auc_provenance.py` separates the three cases:

| condition | AUC |
|---|---|
| raw predictions | 0.768527 |
| **one Platt map applied to every game** | **0.768527** — preserved exactly, to 1e-12 |
| per-window Platt maps, pooled (what the walk-forward does) | 0.758048 (**−0.0105**) |
| isotonic, one map | 0.777474 (+0.0089), 758 distinct values collapsed to 15 |

Two things follow, and both are the reason for the decision:

1. **A strictly monotone map cannot move AUC within a window.** Verified to 1e-12. A bar that requires a calibrator to improve AUC is a bar no calibrator can pass, so it is not a test of anything.
2. **The pooled movement is a cross-window artefact and goes both ways.** Each window is calibrated with its own Platt fit, so the same raw prediction maps to different calibrated values in different windows, and pooling re-orders games *across* windows. On this fixture it moved AUC **down** 0.0105. In the real 758-game run it moved **up** 0.0159. Same mechanism, opposite sign.

A gate that can be passed or failed by a coin flip is not a gate.

Isotonic's movement is a third case and worth naming separately: it is not strictly monotone, so it flattens ties, and AUC scores ties at half credit. Its AUC move is a tie artefact — which is why the doc's isotonic number (+0.0107) is the *smallest* of the two calibrators while its log-loss is the *worst*. Choosing on AUC there would mean choosing the worse model, and the doc already says so.

## What the bar becomes

| | before | after |
|---|---|---|
| AUC gate | 95% interval excludes zero (must improve) | 95% interval excludes a **decline** of any size |

The rest of the rule is unchanged: log-loss, Brier, margin MAE and total MAE still need their intervals to exclude zero in the better direction, and the calibration gap bar (≤ 0.0591) still applies.

This is the *tighter* of the two readings the evidence supports, and deliberately so. The other available reading is to exempt calibrators from the AUC gate entirely. I have not taken it, because it removes a metric from the rule to make something ship, and the honest answer to "did ranking get worse" here is "we cannot tell" — the interval is [−0.0021, +0.0339], which contains a decline.

## Where Platt stands against the new bar

| metric | interval | verdict under the new bar |
|---|---|---|
| win log-loss | [−0.0155, −0.0032] | improved |
| win Brier | [−0.0072, −0.0013] | improved |
| win AUC | **[−0.0021, +0.0339]** | **crosses zero — a decline is not ruled out** |
| margin MAE | [−0.2790, −0.0584] | improved |
| total MAE | [−0.6170, −0.0622] | improved |
| max calibration gap | 0.0581 | below the 0.0591 bar |

**Platt still does not ship.** The tighter bar does not rescue it, because its AUC interval crosses zero in the declining direction as well. It clears the calibration bar, keeps every accuracy metric, and cannot be shown not to have cost ranking. That is the honest read and it is unchanged by the decision.

Isotonic still does not ship either: 2/5 intervals, and log-loss *worse* than baseline at [−0.0147, +0.1014].

## What I am explicitly not doing

- **Not shipping Platt.** The user's decision stands: Platt stays off. This decision changes the *rule*, not the *outcome*, and it would be dishonest to present a rule change as a reason to ship.
- **Not relaxing any other bar.** Only the AUC gate's direction changes, and only because the instrument provably cannot satisfy the old one.
- **Not treating the +0.0159 pooled AUC move as a win.** It is a cross-window re-normalisation, and on the same fixture it moved the other way.

## What would change this decision

- **More holdout games.** At n=758 the AUC interval's half-width is about twice the effect. Everything above is a statement about what can be *detected*, and none of it is a statement that the effect is absent.
- **A calibrator that is not per-window.** If calibration were fitted on a fixed historical set and applied unchanged to every window, the cross-window artefact disappears and the pooled AUC move would be exactly zero — at which point the gate is measuring nothing at all and should be removed outright.
- **A ranking-changing candidate.** For anything that is not a monotone re-expression, the old bar ("must improve") remains the right one, and nothing here relaxes it.

## How the tests pin this

`tests/test_platt_auc_provenance.py` is three tests and each is red-checked:

- `test_one_platt_map_applied_to_all_games_preserves_auc` — asserts one Platt map preserves AUC to 1e-12. If this fails, the monotonicity premise the whole decision rests on is wrong and the decision must be revisited.
- `test_per_window_maps_reorder_games_across_windows` — asserts the within-window order is preserved while the pooled value moves, which is what makes the movement cross-window. Also asserts the direction-independence by printing both, and the fixture is pinned so the −0.0105 is reproducible.
- `test_isotonic_is_not_strictly_monotone_so_it_can_move_auc` — asserts isotonic actually flattens (758 → 15 values), which is the mechanism for its AUC move being a tie artefact.
