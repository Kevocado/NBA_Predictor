# NBA Predictor → NFL/PL Parity — Design Spec

**Date:** 2026-10-04 · **Repo:** `Kevocado/NBA_Predictor` · **Working branch:** `nba-parity-review`
**Parent report:** `~/workspace/goals/sports-predictor-redesign-rollout/files/NBA-predictor-parity-report-2026-10-04.md` (live-site + read-only repo audit vs NFL/PL, 2026-10-04)

## 1. Goal

Bring the NBA predictor to the level of the NFL and PL predictors on **model discipline and evaluation quality**, while preserving the two things NBA already does best: its **track-record honesty machinery** (earliest-pick counting, per-pick provenance, anti-profit copy) and its **injury availability gate** (Out removed, Day-to-Day flagged, 503 fail-closed). The regular season tips off late October 2026 — Phase A must land before it.

Nothing in this spec places bets, publishes, or posts. It produces better models, honest evaluation, and a graded forward-test record.

## 2. Current state (verified 2026-10-04 in this repo — do not re-audit blindly)

### 2.1 Already done (on `nba-parity-review`, 9 commits by Muse, 2026-09-23, never pushed)

These audit gaps are **fixed** — do not redo them:

| Gap (from Oct 4 parity report) | Commit |
|---|---|
| Confidence buckets on track record | `a6494c0` |
| Mean signed error | `a6494c0` |
| Per-position prop MAE | `a6494c0` + `9b2aa0c` + `a6d5917` (position recorded at snapshot) |
| Team logos | `feac67b` |
| 50% break-even marker | `8975dac` |
| Model summary training-data block | `86c19dc` + `13d355d` |
| Missing-model safe skip | `376a749` |

### 2.2 On `origin/main` but not on this branch (merge these — see Phase 0)

- `6bf7316` — **odds refresh loop**: `POST /refresh-odds` now runs on a scheduler thread in the app lifespan (it existed but never ran; Kevin: *"call it on a schedule because nba games are spread throughout the week"*). The deployed container holds `SPORTSBOOK_API_KEY`. **This changes the value-layer design: odds now refresh on a loop — the edge gate and ledger (Phase B) build on this, they don't need to schedule it.**
- `6af77e5` — CI now runs the suite that was never run.
- `5d1766f`, `8dc8be3` — public snapshot refreshes.

### 2.3 Still real gaps (verified against the working tree 2026-10-04)

| # | Gap | Status | Spec section |
|---|---|---|---|
| G1 | `models/evaluate/walk_forward.py` is only `chronological_split` (L4–L14) — a single 80/20 split, not walk-forward. Win accuracy 0.6354 only; no log-loss/Brier/AUC; margin MAE 13.9 / total MAE 18.6 at-or-above naive scale (margin σ≈12, total σ≈15–16) | Open | §5 |
| G2 | Fixed XGBoost ×3 (win classifier + margin/total regressors), `n_estimators=200, max_depth=4`, no candidate race (`src/nba_predictor/models/game_outcome.py` L5–29). Win prob = raw `predict_proba`; margin/total independent regressors, no shared residual distribution | Open | §6 |
| G3 | Model manifest `v20260918120240` — **trained 2026-09-18, 16 days stale**; `manifest_history.jsonl` (4 entries) has **byte-identical metrics** — the daily retrain runs but training data never changes. Suspects: `refresh-data.yml` CI logs since Sep 18; `POST /retrain` (`api/routes.py` L600–614) reading a training-games cache path (`deps.py` L57) the current schedule-cache shape (no box-score fields) can't satisfy. Schedule cache matches the manual `--end 2026-11-30` default, not the workflow's rolling +14d | Open | §4 |
| G4 | Prop models (4 XGBoost regressors, points/rebounds/assists/threes) on **5 rolling means only**, no opponent/rest/usage/position features (`features/player_stats.py` L1–22); evaluation **in-sample only** (`ingest.py` L185–192 discloses this); no chronological holdout at all. Double-double classifier deliberately unserved — keep that decision | Open | §7 |
| G5 | No edge gate: no `threshold`/`min_edge` anywhere, no value-pick endpoint. Edges stored and rendered, nothing selects. No graded value-bet ledger, no CLV | Open | §8 |
| G6 | No snapshot staleness gate: Oct-4 snapshot happily serves Sep-18 models (`public_snapshot.py`) | Open | §9 |
| G7 | Dead code: `data/odds_api.py` (The Odds API, unused — pipeline imports only `sportsbook_api`, `pipeline/refresh_odds.py` L4), `data/balldontlie.py`, `data/nba_api.py` (no production caller); `get_player_props` written against a schema never observed live; writer-less tracking tables `odds_timing_snapshots`, `game_forecast_snapshots` (`tracking/store.py` L99–118) | Open | §10 |
| G8 | `features/injuries.py` default path neutralised after the LeBron-fabrication incident (correct) — but leaves **no injury feature** | Open | §6 |
| G9 | Preseason: no season-state flag; `playoff_status` seeded by raw win% prints "Clinched"/"eliminated" from empty records (`ingest.py` L422–467); features causal via shift(1) but no preseason flag (upcoming = not completed) | Open | §11 |
| G10 | Frontend dead labels: `METRIC_NAMES` (`ModelSummaryPage.tsx` L12–L20 per audit) lists log_loss/brier/AUC labels the backend never emits — emit them (§5) or remove the labels | Open | §11 |
| G11 | Game cards: no game status (FINAL/HIT-MISSED/LIVE/NEXT UP), no pick-timing badges on cards, no sort options; detail lacks model-vs-line with edge, uncertainty (±), injury line; no per-pick detail table; no timezone label on tip times | Partially open | §11 |
| G12 | Ops: `deploy-azure.yml` is manual-only but README claims every push redeploys — contradict each other | Open | §12 |

### 2.4 Things that must not regress

- Earliest-pick counting, `pre_kickoff` subset, per-pick provenance, pushes never scored as misses, `hit_rate: None` for ungradeable markets, anti-profit copy in the vs-market block.
- Injury gate: Out removed (surfaced separately), Day-to-Day flagged-not-removed, 503 fail-closed if feed unreadable. Fabricated-injury strings pinned absent (`tests/test_data_injuries.py`). **No injury data enters features unless sourced from the verified ESPN availability feed.**
- The double-double classifier stays unserved.

## 3. Architecture decisions (copied from the sibling repos — do not reinvent)

1. **Candidate race, then freeze the winner** (NFL pattern): Elo/Ridge/XGBoost race on walk-forward log-loss; NFL's Ridge *beat* XGBoost (`models/game_outcome.py` L36–53, manifest `"chosen_candidate": "ridge"`). NBA runs the same race; the manifest records the winner. Do not assume XGBoost wins — NBA's margin/total MAE suggests it's overfitting noise on 24 features.
2. **One residual-σ probability model** (NFL `game_outcome.py` L66–111): fit one Normal residual σ for margin and one for total; win/cover/over come from the CDF so they're coherent. Cover only when a **real** line exists (the explicit fix for CFB's fake-0.5 bug).
3. **Expanding-window walk-forward** (NFL `evaluate/walk_forward.py` L212–249; PL Optuna-over-5-fold-walk-forward): the single 80/20 split is replaced. Report log-loss, Brier, AUC vs naive; margin/total MAE vs naive scale.
4. **Calibration stance** (NFL `calibration.py`): measurement-only reliability curves (4 buckets, pre-game snapshots); **explicit no-Platt/isotonic policy**. PL rejected Platt scaling after it worsened Brier/log-loss. NBA keeps its display-only reliability bins (`services/calibration_service.py`).
5. **Offline artifact gate** (NFL `models/training.py` L101–119): refuse to write a prop-model artifact if it can't clear a calibration check (±5pt/bucket at n≥100) on held-out data. This is what stops in-sample-metric models from being served.
6. **The sibling value contract** (NFL `build_value_bet_table`, `edge_threshold=0.05`; PL 5% × data-confidence, min prob 0.30, max odds 6.0, **odds ≤1h old**): 5% edge threshold, at most one single per game, no parlays, odds-freshness cap, same-book/same-point pairing (NFL's post-review fix). Settle every flagged pick; track **CLV per pick** (NFL `store.py` L1836–64; PL `_clv_pct`, `update_closing_lines`). Copy PL's honest "not evidence of a profitable strategy" framing.
7. **Snapshot staleness gate** (NFL `public_snapshot.py` L226–272): refuse to publish `public_snapshot.json` when models are older than X. NBA currently serves Sep-18 models under an Oct-4 snapshot.
8. **Additive artifacts**: new models are versioned (`models/*_v2_*.pkl`, manifest history appended); production pickles are never overwritten until Kevin approves. Same convention as the NFL props rebuild (`models/*_quantile_2025.pkl`).

## 4. Phase A1 — Diagnose the stalled retrain

**Why first:** every phase below inherits a frozen dataset until this is fixed. One CI-log check, then the fix.

- Read `refresh-data.yml` CI logs since Sep 18: does the training-games pull succeed? Is the cache written where `POST /retrain` (`api/routes.py` L600–614) looks (`deps.py` L57)?
- Check whether the schedule cache window matches the workflow's rolling +14d (it currently matches the manual `--end 2026-11-30` default — consistent with a frozen pipeline).
- Fix: retrain must ingest fresh training games and `manifest_history.jsonl` must show moving metrics. If box-score fields are missing from the schedule cache, either add them to the cache build or point retrain at a cache that has them.
- **Done when:** a manual retrain produces a manifest with a new `trained_at` and changed (not byte-identical) metrics, and the daily workflow keeps it moving. Write a test asserting `manifest_history.jsonl` entries differ across two retrains on different data (the "never-frozen" regression test).

## 5. Phase A2 — Real walk-forward evaluation + naive baselines

Replace `models/evaluate/walk_forward.py`'s `chronological_split` helper with an expanding-window walk-forward:

- **Win model:** log-loss, Brier, AUC per window and pooled, vs naive (home-win base rate / 0.6931 coin-flip). NFL bar: 0.6227 vs 0.6931.
- **Margin/total:** MAE per window vs naive scale (previous-season margin σ≈12, total σ≈15–16; and vs a naive "home −3 / total = league avg" baseline). Current 13.9/18.6 must be shown against these baselines honestly.
- Keep `chronological_split` available (tests use it) but stop calling it "walk-forward" anywhere user-facing.
- Emit `log_loss`, `brier`, `auc` from the backend into the manifest (this also fixes G10: the frontend labels stop being dead).
- **Done when:** the evaluation script prints a table of metric vs naive for each model, committed as a markdown report in `docs/` (honest numbers, whatever they are).

## 6. Phase A3 — Candidate race + coherent probability math + features

- **Race:** Elo baseline / Ridge / XGBoost on expanding-window walk-forward log-loss for win probability; Ridge / XGBoost for margin and total on MAE. Record the winner in the manifest (`"chosen_candidate"`). Retire the loser from the serving path (keep the code).
- **Probability math:** one fitted Normal residual σ per output (margin, total); win/cover/over via CDF. Cover only when a real line exists (sportsbook feed, not a placeholder).
- **Features (game side):** add pace, opponent-adjusted ratings, rest-days delta, and injury deltas derived **only** from the verified ESPN availability feed (`api/availability.py` data — never fabricate; the LeBron-incident neutralisation stays). All features shift(1)-then-rolling — a game-G feature may only use games < G.
- **Done when:** walk-forward log-loss beats naive (or the honest report says it doesn't), margin/total MAE improves vs §5 baseline, and the manifest's `chosen_candidate` + residual σs drive the served probabilities.

## 7. Phase A4 — Props: holdout, features, artifact gate

- **Holdout:** chronological holdout for all four prop regressors (points/rebounds/assists/threes). No more in-sample-only metrics.
- **Features:** opponent-defense-vs-position (à la NFL's 26-feature set), rest days, usage/minutes trend, position-keyed treatment (G/F/C — maps onto NFL's QB/RB/TE/WR groups). Keep shift(1)-then-rolling; no leakage.
- **Artifact gate:** port NFL's offline calibration gate concept — a prop model that can't clear a calibration check on held-out data does not get its artifact written.
- **Ledger tables already exist:** `game_player_outcomes` + `player_prediction_snapshots` (`tracking/store.py` L23–87) need writers/readers (per-player graded prop ledger). Position is now recorded at snapshot (`9b2aa0c`, `a6d5917`).
- **Done when:** held-out MAE reported per market and per position, gate passes or honestly fails, and the per-player ledger has writers + readers with tests.

## 8. Phase B1 — Edge gate (builds on the odds-refresh loop)

- `POST /refresh-odds` now runs on a loop (PR #32) — **do not add another scheduler**.
- Implement the sibling contract: **5% edge threshold**, at most one single per game, no parlays, **odds ≤1h old** (stale odds → zero flags), same-book/same-point pairing for totals.
- A value-pick endpoint (or section of the existing odds payload) that returns only gated picks.
- **Done when:** with the loop running, the endpoint returns gated picks for a slate with fresh odds and zero flags for a slate with stale odds (test pins both).

## 9. Phase B2 — Graded value ledger + CLV

- Track every flagged pick: snapshot (pick, line, odds, book, model prob, edge, timestamp — immutable, pre-tip), settle vs actual, CLV per pick vs closing line (`update_closing_lines` pattern).
- Copy PL's honest framing: the ledger shows yield **whatever it is**, with "not evidence of a profitable strategy" copy. This is a measurement tool, not a profit claim.
- **Done when:** a weekly markdown report renders the ledger (picks, hit rate, yield, CLV), committed to `docs/` or served from the tracking DB.

## 10. Phase B3 — Staleness gate + dead-code cleanup

- **Staleness gate:** port NFL `public_snapshot.py` L226–272 — refuse to publish when models are older than X (X = 14 days suggested; Kevin decides).
- **Dead code** — delete or wire, no third option:
  - `data/odds_api.py`: delete (pipeline uses `sportsbook_api`; PL demoted The Odds API to fallback for quota reasons, but NBA's module was never wired and its replaced predecessor called a fabricated path — delete it).
  - `data/balldontlie.py`, `data/nba_api.py`: delete (no production caller; ESPN keyless is the only real source).
  - `get_player_props`: validate against the live sportsbook schema or delete.
  - `odds_timing_snapshots`, `game_forecast_snapshots` tables: use or drop.
- **Done when:** `grep` finds no dead module imports, and a snapshot publish with a 20-day-old model is refused by test.

## 11. Phase C — Frontend polish + preseason + ops truth

- **Game cards:** game status (FINAL + HIT/MISSED, LIVE, NEXT UP), pick-timing badges on cards, sort options (kickoff order / most confident first).
- **Game detail:** model-vs-line block with edge, projected margin with uncertainty (±), injury-report line.
- **Track record:** per-pick detail table (sortable: game, market, pick, actual, hit/miss, when made), biggest-upsets/misses table (PL pattern), projected final standings (PL's ProjectedTable).
- **Timezone:** label tip times (NFL: "Kickoff times in UTC").
- **Preseason flag:** season-state flag (offseason/preseason/regular/postseason); suppress or relabel `playoff_status` and projections during offseason (no more "Clinched" from empty records).
- **Dead labels:** verify `METRIC_NAMES` — §5 makes the backend emit log_loss/brier/AUC, so the labels should light up; remove any that stay dead.
- **Ops truth:** `deploy-azure.yml` is manual-only while README claims auto-deploy — either restore push-triggered VPS deploy (NFL/PL `deploy.yml` → GHCR → `deploy <sport> <sha>` with health-check + rollback) or fix the README. Kevin decides.

## 12. Hard-won conventions (from `~/AGENTS.md` — these bit us before)

1. **Stub verification:** when a test stubs an external API, verify the stub against the real API's signature — a wrong stub makes the test agree with the bug (2026-10-04: a stub had the same wrong shape as the bug, the pull silently fetched nothing).
2. **All-NaN columns = broken pull until proven otherwise:** a feature column of all-NaN reads as "no signal" — treat unexpected all-NaN as a broken pull (2026-10-04: Open-Meteo **forecast** endpoint only serves ~3 months; 411 misses looked like no weather signal until the **archive** endpoint returned 1,257 readings). If weather features are added, use the archive endpoint for training.
3. **shift(1)-then-rolling, no exceptions.** A rolling feature that includes the target game silently inflates walk-forward scores.
4. **No injury fabrication, ever.** Neutralise-and-pin is the precedent (`NotImplementedError` + pinning test). Injury features come only from the verified ESPN availability feed.

## 15. Ponytail — the anti-over-engineering rule (binding)

Development on this spec uses the **ponytail** skill (DietrichGebert/ponytail, MIT — the "laziest senior developer" skill) together with the superpowers TDD workflow. If the skill is installed in the agent's environment, use it. If not, follow the ladder below inline — it is part of this spec.

**The 7-rung ladder — climb down before writing any code, stop at the first rung that holds:**

1. **YAGNI** — Does this code/abstraction need to exist right now? If not, don't write it. No speculative generality, no "framework for future models."
2. **Reuse existing code** — An identical or near-identical function/component already exists? Reuse and compose. This spec's architecture decisions (§3) are explicit reuse directives: the candidate-race pattern, residual-σ math, the 5% gate contract, and the staleness gate are copied from the NFL/PL repos, not reinvented.
3. **Standard library** — `pathlib`/`itertools`/`statistics`, `scipy.stats.norm.cdf`, `sklearn` metrics — before custom code.
4. **Native platform features** — HTML5/CSS over JS where it holds; SQLite (already the tracking DB) over any new store.
5. **Existing dependencies** — Check `pyproject.toml` / `package.json` first. **Do not install new packages.** XGBoost, scikit-learn, scipy, pandas, FastAPI are already there; they cover everything in this spec.
6. **One-liner** — If the logic fits in a single readable line or a small pure function, that's the implementation.
7. **Minimum viable code** — Otherwise, the absolute minimum diff that fulfills the task and passes its tests. Zero dead code, zero speculative parameters.

**How ponytail meets this spec's phases:**
- Phase A reuses NFL/PL evaluation and model-selection patterns verbatim (rung 2) instead of designing new ones.
- Phase B's edge gate is a small pure function over the odds payload the refresh loop already produces (rungs 2, 6, 7) — no new scheduler, no new service.
- Phase B3/B4 is deletion (rung 1 applied to the codebase itself): dead modules and unused tables go.
- Phase C copies PL/NFL frontend patterns (rung 2) rather than inventing components.

TDD is not relaxed by ponytail: every task still starts with a failing test. Ponytail decides *how little* code the test needs; TDD decides *whether* it's correct.

## 16. Impeccable — the frontend skill (binding for all UI work)

All frontend work in this spec (Phase C, and any UI touch in other phases) uses the **impeccable** skill (pbakaus/impeccable, Apache 2.0) — the design skill for AI coding agents that extends Anthropic's frontend-design skill with 24 commands and 61 deterministic detector rules for AI design mistakes.

- **Install (once per agent environment):** `npx impeccable install`, then `/impeccable init` in the repo root — `init` inspects the project and writes `PRODUCT.md` (audience, purpose, constraints, voice). Commit `PRODUCT.md`; later commands use it as context.
- **Commands to use:** `/impeccable craft` (plan then build a new surface), `/impeccable polish` (final pass before the phase PR), `/impeccable audit <surface>` (accessibility, performance, responsiveness checks), `/impeccable critique` (UX review of hierarchy/clarity/tone), `/impeccable harden` (error states, text overflow, edge cases).
- **Binding rule:** every frontend task ends with `/impeccable audit` on each touched surface, and detector findings are fixed before the commit — not after the phase.
- **If the skill is not installed:** fall back to the repo's incumbent visual system — the dark "PREDICTOR" theme, existing CSS variables (`--color-net-*`, `--color-line`, etc.), and the shared `predictor-ui` components. No new design language, no new fonts, no new color system. Match what's there.

Impeccable governs *how the UI looks*; ponytail governs *how little code builds it*; TDD governs *whether it works*. All three apply to frontend tasks.

## 13. Global constraints (every task inherits these)

- **$0 data spend.** No paid APIs, no new keys. ESPN keyless API and RapidAPI Sportsbook API v2 (already keyed, on a refresh loop) are the only live calls. The Odds API is deleted per §10 — it is not a fallback here.
- **Additive changes only.** New models are versioned (`models/*_v2_*.pkl`, manifest history appended); production pickles are never overwritten until Kevin approves. The live site keeps working throughout; no schema change may break existing readers (migrations use ALTER TABLE ADD COLUMN).
- **Snapshots are immutable** (INSERT OR IGNORE); reconciliation only fills outcome columns. Predictions snapshotted strictly pre-tip-off. Earliest-pick counting is never displaced by reruns.
- **No injury fabrication.** §12.4 is absolute.
- **Nothing publishes, posts, or places bets.** Model + honest evaluation + graded ledger, full stop.
- **Ponytail + TDD:** every task starts with a failing test and climbs the ponytail ladder (§15) before any code is written — reuse before inventing, minimum viable diff. Commit after every task.
- **Review gates:** Phase A ends with the honest evaluation report (§5) — Kevin reviews before Phase B starts. Phase B ends with the ledger's first weekly report. No phase starts without Kevin's go-ahead.
- **Never merge to `main`** without Kevin's explicit approval. Work on `nba-parity-review`; open a PR when a phase is done.

## 14. Review focus (failure modes most likely to bite)

1. **Leakage via future games** — a rolling feature including the target game inflates walk-forward scores. Pinned by the shift(1) test in the plan; re-verify by asserting no feature timestamp ≥ target game date.
2. **Frozen training data** — the retrain runs but ingests nothing new (the Sep 18 bug). Pinned by the "never-frozen" regression test: two retrains on different data must produce different metrics.
3. **Post-tip-off snapshots** — a line snapshot recorded after tip-off is not a bettable price. The odds loop runs continuously; the gate must reject stale/after-tip odds (test pins the ≤1h freshness rule).
4. **Player-ID mismatch** between ESPN (player IDs) and the sportsbook props feed (names) — join on normalized (name, team); unmatched props are logged and skipped, never guessed.
5. **Quantile/crossing and NaN features** — if quantile models are used, enforce monotonicity; if a feature column is all-NaN, the pull is broken (§12.2), not the feature.
