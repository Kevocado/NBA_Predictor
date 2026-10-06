# NBA Predictor — Phase A honest evaluation

**Date:** 2026-10-05 · **Branch:** `nba-parity-review` · **Spec:** `docs/superpowers/specs/2026-10-04-nba-parity-design.md`

This is the Phase A review gate. It reports what the models actually do on
out-of-fold data against naive baselines, including where they lose. Nothing
here was tuned to look better. Two of the three models **do not beat naive**,
and the served win model was **18 points overconfident** on its most confident
picks before this work.

## Method

- **1,368 games** from `data/cache/training/games.json` (2025-10-02 → 2026-10-04),
  24 features from `features/build.py`.
- **Expanding-window walk-forward**, windows cut on **date boundaries** (the NBA
  plays 10–12 games a night; a row-wise cut splits a date and leaks). Every
  window refits from scratch on its own training slice.
- **Naive baselines are leak-free**: the home-win base rate comes from each
  window's *training* data, never its test labels.
- Pooled figures concatenate every out-of-fold prediction (n=758).

## Headline: the win model is miscalibrated, and it was the model's fault

Kevin's read that the model is badly calibrated is confirmed, and quantified.

| model | log-loss | Brier | AUC | max calibration gap |
|---|---|---|---|---|
| **logistic (race winner)** | **0.6636** | **0.2351** | **0.6285** | **0.0592** |
| xgboost (was served) | 0.7059 | 0.2516 | 0.5912 | 0.1831 |
| naive — always home | 0.6864 | 0.2466 | 0.500 | n/a |
| coin flip | 0.6931 | 0.2500 | 0.500 | n/a |

**The incumbent XGBoost lost to both naive baselines** — worse than always
picking the home team. Its calibration was the worst part: bucketed into
quintiles of predicted probability, its most confident bucket predicted **0.80**
and actually won **0.618**. Eighteen points of overconfidence on exactly the
picks a visitor would act on. That is a real defect, not a rounding concern.

The race winner cuts that gap by two thirds (0.1831 → 0.0592) and is the first
win model to beat naive on log-loss, Brier and AUC at once.

**It still misses the ±0.05-per-bucket calibration bar** at 0.0592. Under the
NFL-style gate in spec section 10 this model would be **refused**. That is
reported, not worked around.

## The race reversed the core assumption (gap G2)

The three models were fixed XGBoost by assumption. Walk-forward says otherwise.

| target | winner | winner's score | rival | naive (mean) | naive (fixed) |
|---|---|---|---|---|---|
| win | **logistic** | **0.6636** | xgboost 0.7059 | 0.6864 | 0.6931 (coin-flip) |
| margin | **ridge** | **13.232** | xgboost 13.631 | 14.018 | 13.933 (−3) |
| total | **ridge** | **16.512** | xgboost 17.110 | 16.375 | 16.458 (league avg) |

**NFL precedent held.** The NFL repo's Ridge beat XGBoost at this sample size,
and so does a plain `LogisticRegression` here. A 200-tree depth-4 booster on 24
features and 1,368 games was overfitting noise, which is what gap G2 suspected.

## Honest losses

**1. Total MAE still loses to naive, on both baselines.** Ridge 16.512 against
16.375 (per-window mean) and 16.458 (league average). The margin and total
regressors are not earning their complexity. Two of three targets beat naive;
total does not. No gate was weakened to change this.

**2. The margin model's margin over naive is thin.** 13.232 vs 13.933 (−3) is a
real 0.7-point gain; 13.232 vs 14.018 (mean) is 0.8. Worth having, not a large
edge.

**3. Elo beats every ML candidate.** A plain rating loop scores **0.6037**
log-loss — better than the race winner's 0.6636. Spec section 3.1 designates
Elo reference-only and never served, so it is **not** in the serving path and
was not promoted. But "our model loses to arithmetic by six points of log-loss"
belongs in front of Kevin, and it is a question about whether a rating book
should serve the win market at all.

**4. Walk-forward costs more than the old split, and the numbers differ.**
Scored on the 80% chronological split the win log-loss reads 0.7712 with AUC
0.5496; on the full frame it reads 0.6636 with AUC 0.6285. Same model, same
method, different data. The pooled full-frame figure is the one quoted
throughout because it uses every game.

**5. Accuracy fell from 0.6354 to 0.5844 under honest evaluation.** The old
figure came from a single 80/20 `chronological_split`. Walk-forward accuracy on
out-of-fold data is 0.5844 — about 5 points lower for the same model. This is
the single most valuable number in the report: it is why the spec demanded
re-evaluation, and it means the previously published accuracy was flattering.

## Residual σ and coherent probabilities

One Normal residual σ per output, so win/cover/over cannot disagree —
`cover_prob(m, 0, σ) == win_prob(m, σ)` exactly, because covering a pick'em line
*is* winning. `cover_prob` returns `None` with no real line rather than
fabricating a placeholder (CFB's bug, one level up).

Fitted from out-of-fold residuals: **margin σ = 16.58, total σ = 20.69**, against
the spec's naive references of σ≈12 and σ≈15–16. Larger because the model's real
error is worse than naive; using the naive σ would make every probability read
more confident than the evidence supports.

## What changed underneath

**The retrain had never succeeded.** All 17 `refresh-data.yml` runs since
2026-09-18 failed. The workflow's rolling `60 days ago` window could not cover a
season, so in October `to_training_frame` returned **one** game and
`chronological_split` raised before any manifest was written. The commit step
sat downstream, so nothing was ever written — `manifest_history.jsonl` held four
byte-identical entries. Fixed with a season-anchored window, plus a write for
`data/cache/training/games.json`, which `POST /retrain` read and **no code in
`src/` ever produced**, making that endpoint unrunnable everywhere including the
VPS.

## Gates, honestly

| gate | result |
|---|---|
| win log-loss beats naive | **PASS** (0.6636 vs 0.6864) — first time |
| calibration ±5pt/bucket at n≥100 | **FAIL** (0.0592 vs 0.05) |
| margin MAE beats naive | **PASS** (13.232 vs 13.933) |
| total MAE beats naive | **FAIL** (16.512 vs 16.375) |
| margin/total MAE improved vs §5 baseline | **MARGINAL** — total still ≥ naive |
| manifest emits log_loss/brier/auc | **PASS** — G10 labels now live |

**Two gates fail. Both are reported as failures.** No gate was relaxed to make
it pass, and the numbers above are reproducible from the committed code.

## Open questions for Kevin

1. **Serve Elo?** It outscores the model by 6 points of log-loss. Spec says
   reference-only; changing that is a spec amendment, not an implementation
   choice.
2. **The calibration gate at ±0.05 refuses the current winner.** Options: accept
   the refusal and ship nothing, loosen to ±0.06, or hold the model until Task 7
   features are in and re-race. My read is the third — the new game-context
   features are built but **not yet in the served model**, and they are the
   plausible route to a real calibration fix.
3. **Total MAE.** If it still loses to naive after features, the honest move is
   to serve the league-average total and say so.

## Reproducing

```
python3 -m pytest tests/ -q          # 689 passed, 21 skipped
```

Walk-forward: `src/nba_predictor/models/evaluate/walk_forward_eval.py`.
Race: `src/nba_predictor/models/candidate_race.py` — a candidate that raises
**fails the race** rather than being scored `inf` and losing silently, which is
what produced a fake "XGBoost wins" result in a rejected draft.
Probabilities: `src/nba_predictor/models/probability.py`.