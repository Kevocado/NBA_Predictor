# NBA Predictor → NFL/PL Parity — Implementation Plan

> **For agentic workers:** REQUIRED: follow this plan task-by-task in order.
> Use the **ponytail skill** (DietrichGebert/ponytail — the anti-over-engineering
> ladder; spec §15 states the ladder inline so it holds with or without the
> skill installed) together with superpowers test-driven development: write the
> failing test first for every task, no production code without a failing test,
> and climb the ladder before writing any code — reuse before inventing, stdlib
> before custom, one line before fifty, minimum viable diff always. **For every
> frontend task, also use the impeccable skill** (pbakaus/impeccable; spec §16):
> `/impeccable craft` or `polish` to build, `/impeccable audit` on each touched
> surface before committing, and fix detector findings in the same task. Steps use
> checkbox (`- [ ]`) syntax for tracking. Commit after every task. Never merge
> to `main` without Kevin's explicit approval — work on `nba-parity-review` and
> open a PR per phase.

**Goal:** Bring the NBA predictor's model discipline and evaluation quality to the NFL/PL level — real walk-forward evaluation with naive baselines, a candidate race that picks the winner, coherent probability math, held-out prop models behind an artifact gate, a 5%-edge gate with a graded value ledger and CLV — without touching the honesty machinery or the injury gate.

**Architecture:** Extend the repo in place on `nba-parity-review` (which already holds the Sep 23 review-implementation commits). Phase 0 merges `origin/main` (odds-refresh loop, CI). Phase A fixes the model pipeline (retrain → evaluation → candidate race → props). Phase B adds the value layer (edge gate, ledger, staleness gate, dead-code removal). Phase C is frontend polish, preseason handling, and ops truth. New model artifacts are versioned (`models/*_v2_*.pkl`); production pickles are never overwritten until Kevin approves.

**Tech Stack:** Python ≥3.13, pandas, numpy, scikit-learn (Ridge), XGBoost, scipy (Normal CDF), FastAPI, SQLite (existing tracking DB), pytest (`pythonpath=["src"]`), React/TypeScript frontend.

**Spec:** `docs/superpowers/specs/2026-10-04-nba-parity-design.md` — the plan argues from the spec; read both before starting.

## Global Constraints

- $0 data spend. No paid APIs, no new keys. ESPN keyless API and RapidAPI Sportsbook API v2 (already keyed, on a refresh loop since PR #32) are the only live calls. `data/odds_api.py` is deleted in Task 15 — it is not a fallback.
- Additive changes only: new pickles are `models/*_v2_*.pkl` with manifest history appended; production pickles are never overwritten until Kevin approves.
- The live predictor site keeps working throughout. No schema change may break existing readers — migrations use ALTER TABLE ADD COLUMN.
- Every rolling feature uses shift(1)-then-rolling — a game-G feature may only use games < G. No exceptions.
- Snapshot rows are immutable (INSERT OR IGNORE); reconciliation only fills outcome columns. Predictions snapshotted strictly pre-tip-off. Earliest-pick counting is never displaced by reruns.
- No injury fabrication, ever (spec §12.4). Injury features come only from the verified ESPN availability feed.
- Nothing publishes, posts, or places bets. Model + honest evaluation + graded ledger, full stop.
- Review gates: Phase A ends with the honest evaluation report (Task 11) — Kevin reviews before Phase B. Phase B ends with the ledger's first weekly report. No phase starts without Kevin's go-ahead.

## Review Focus

1. **Leakage via future games** — a rolling feature that includes the target game silently inflates walk-forward scores. Pinned by the shift(1) test (Task 7) and re-verified by asserting no feature timestamp ≥ target game date.
2. **Frozen training data** — the retrain runs but ingests nothing new (the Sep 18 bug). Pinned by the never-frozen regression test (Task 2): two retrains on different data must produce different metrics.
3. **Post-tip-off snapshots** — a line snapshot recorded after tip-off is not a bettable price. The odds loop runs continuously; the gate must reject stale/after-tip odds (Task 12 pins the ≤1h freshness rule).
4. **Player-ID mismatch** between ESPN (player IDs) and the sportsbook props feed (names) — join on normalized (name, team); unmatched props are logged and skipped, never guessed (Task 10).
5. **All-NaN feature columns** — an all-NaN column is a broken pull, not "no signal" (spec §12.2). Task 7's feature tests assert non-NaN coverage thresholds per feature.

---

## File structure

New files:
- `src/nba_predictor/models/evaluate/walk_forward_eval.py` — expanding-window walk-forward for win/margin/total (Tasks 3–4)
- `src/nba_predictor/models/candidate_race.py` — Elo/Ridge/XGBoost race + manifest winner recording (Task 5)
- `src/nba_predictor/models/probability.py` — residual-σ Normal-CDF win/cover/over math (Task 6)
- `src/nba_predictor/features/game_context.py` — pace, opponent-adjusted ratings, rest, injury deltas (Task 7)
- `src/nba_predictor/models/prop_holdout.py` — chronological holdout evaluation for the 4 prop regressors (Task 8)
- `src/nba_predictor/features/prop_matchup.py` — opponent-D-vs-position, rest, usage/minutes, position-keyed features (Task 9)
- `src/nba_predictor/models/artifact_gate.py` — offline calibration gate for prop artifacts (Task 10)
- `src/nba_predictor/tracking/value_ledger.py` — graded value-bet ledger + CLV (Task 13)
- `src/nba_predictor/odds/edge_gate.py` — 5% gate, one single/game, freshness, pairing (Task 12)
- `docs/nba-parity-evaluation-2026-10.md` — honest evaluation report (Task 11)

Modified files:
- `src/nba_predictor/models/evaluate/walk_forward.py` — keep `chronological_split`, stop calling it walk-forward (Task 3)
- `src/nba_predictor/models/game_outcome.py` — race winner serves; residual-σ probs (Tasks 5–6)
- `src/nba_predictor/tracking/store.py` — ledger writers/readers; drop or use dead tables (Tasks 10, 13, 15)
- `src/nba_predictor/public_snapshot.py` — staleness gate (Task 14)
- `src/nba_predictor/api/routes.py` — retrain fix, value-pick endpoint (Tasks 2, 12)
- `src/nba_predictor/pipeline/*` — retrain data flow fix (Task 2)
- `models/manifest.json` — `chosen_candidate`, residual σs, log_loss/brier/auc emission (Tasks 4–6)
- `frontend/src/**` — cards, detail, track record, preseason flag, dead labels (Tasks 16–21)

Deleted files (Task 15): `src/nba_predictor/data/odds_api.py`, `src/nba_predictor/data/balldontlie.py`, `src/nba_predictor/data/nba_api.py` (+ their tests `tests/test_data_odds_api.py`, `tests/test_data_balldontlie.py`, `tests/test_data_nba_api.py` if they only cover the dead modules).

---

## Phase 0 — Reconcile branches (do this first)

### Task 0: Merge `origin/main` into `nba-parity-review`, run the suite, push

**Files:** repo root (merge), `frontend/package-lock.json` (modified, include in commit)

**Context:** This branch holds 9 unpushed review commits (2026-09-23). `origin/main` is 5 commits ahead: PR #32 (odds-refresh loop — another agent, opencode, 2026-10-04), a CI fix, and snapshot refreshes. The odds-refresh loop changes the value-layer design: `POST /refresh-odds` now runs on a scheduler thread in the app lifespan — Task 12 builds the gate on it, it does not schedule anything.

- [ ] **Step 1: Fetch and merge**

```bash
git fetch origin
git merge origin/main -m "merge: origin/main (odds-refresh-loop, CI) into nba-parity-review"
```

- [ ] **Step 2: Resolve conflicts** — keep both features. Likely touchpoints: `src/nba_predictor/tracking/store.py`, `src/nba_predictor/api/routes.py`, snapshot payloads. When in doubt, keep the review-branch version of honesty/timing logic and the main-branch version of the odds loop.
- [ ] **Step 3: Run the full suite**

```bash
python -m pytest tests/ -x -q
```

Expected: all pass. If the suite fails on a conflict you introduced, fix it — do not delete tests to make it green.
- [ ] **Step 4: Verify the odds loop survived the merge** — start the app locally and confirm the refresh-odds scheduler thread is registered in the lifespan (log line or startup print). If it's missing, the merge dropped it — restore from `6bf7316`.
- [ ] **Step 5: Commit and push the branch**

```bash
git add -A
git commit -m "chore: merge origin/main into nba-parity-review; include package-lock"
git push -u origin nba-parity-review
```

Expected: branch visible on GitHub for other agents to pull.

---

## Phase A — Model pipeline (P0)

### Task 1: Diagnose the stalled retrain

**Files:**
- Read: `.github/workflows/refresh-data.yml`, `src/nba_predictor/api/routes.py` (L600–614), `src/nba_predictor/api/deps.py` (L57), `models/manifest_history.jsonl`

**Interfaces:** none yet — this task produces a diagnosis written into Task 2's fix.

- [ ] **Step 1: Read the CI logs** — open the last 3 runs of `refresh-data.yml` (GitHub Actions UI or `gh run list --workflow refresh-data.yml`). Record: did the training-games pull succeed? How many games were written? Where?
- [ ] **Step 2: Trace the retrain path** — read `api/routes.py` L600–614 (`POST /retrain`) and `deps.py` L57 (the training-games cache path it reads). Compare against the cache the schedule build actually writes (check the `--end` default vs the workflow's rolling +14d — the cache currently matches the manual `--end 2026-11-30` default, not the workflow window).
- [ ] **Step 3: Check the cache shape** — does the training-games cache contain the box-score fields the retrain needs? (`No box-score fields` was the suspect.)
- [ ] **Step 4: Write the diagnosis** — a short markdown note (commit it at `docs/nba-retrain-diagnosis-2026-10.md`): root cause in one paragraph, the exact file/line where data stops flowing, and the fix Task 2 will implement.
- [ ] **Step 5: Commit**

```bash
git add docs/nba-retrain-diagnosis-2026-10.md
git commit -m "docs: retrain stall diagnosis"
```

### Task 2: Fix the retrain data flow + never-frozen regression test

**Files:**
- Modify: `src/nba_predictor/api/routes.py` (retrain endpoint), `src/nba_predictor/pipeline/*` (cache build — whichever file Task 1 identified), `.github/workflows/refresh-data.yml` (if the window is wrong)
- Test: `tests/test_retrain_freshness.py`

**Interfaces:**
- Produces: `POST /retrain` writes `models/manifest.json` with new `trained_at` and appends a non-identical entry to `models/manifest_history.jsonl`.

- [ ] **Step 1: Write the failing test** — the never-frozen regression test:

```python
def test_two_retrains_on_different_data_produce_different_metrics(tmp_path):
    # Build two tiny training sets that differ (e.g. different game dates).
    # Run the retrain entrypoint twice, once per set.
    # Assert manifest_history.jsonl's last two entries differ in at least one metric.
    m1 = run_retrain(training_games=games_a, manifest_path=tmp_path / "manifest.json",
                     history_path=tmp_path / "history.jsonl")
    m2 = run_retrain(training_games=games_b, manifest_path=tmp_path / "manifest.json",
                     history_path=tmp_path / "history.jsonl")
    assert m1["metrics"] != m2["metrics"], "retrain is frozen: metrics byte-identical across different data"
```

Run: `pytest tests/test_retrain_freshness.py -v` — Expected: FAIL (helper `run_retrain` doesn't exist yet, or metrics identical).
- [ ] **Step 2: Implement the fix** from Task 1's diagnosis — the minimal change that makes training games actually flow into the retrain (fix the cache path, add box-score fields to the cache build, and/or correct the workflow window to rolling +14d).
- [ ] **Step 3: Run the test** — Expected: PASS.
- [ ] **Step 4: Manual retrain** — run `POST /retrain` (or the CLI entrypoint) once against real data; confirm `models/manifest.json` shows a new `trained_at` and today's date, and `manifest_history.jsonl` has a new, non-identical entry.
- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/api/routes.py src/nba_predictor/pipeline/ tests/test_retrain_freshness.py .github/workflows/refresh-data.yml docs/nba-retrain-diagnosis-2026-10.md
git commit -m "fix: retrain ingests fresh training games; add never-frozen regression test"
```

### Task 3: Expanding-window walk-forward for the win model

**Files:**
- Create: `src/nba_predictor/models/evaluate/walk_forward_eval.py`
- Modify: `src/nba_predictor/models/evaluate/walk_forward.py` (keep `chronological_split`; remove any user-facing "walk-forward" naming tied to it)
- Test: `tests/test_walk_forward_eval.py`

**Interfaces:**
- Consumes: game frame with `game_date`, features, `home_win` label; model factory `fn(train_df) -> predict_proba_fn`.
- Produces: `walk_forward_metrics(df, model_factory, windows) -> dict` with `log_loss`, `brier`, `auc` per window + pooled, and `naive_log_loss` (home-win base rate), `coinflip_log_loss` (0.6931).

- [ ] **Step 1: Write the failing test**

```python
def test_walk_forward_windows_are_expanding_and_causal():
    df = make_games(n=100)  # game_date sorted, home_win label
    res = walk_forward_metrics(df, model_factory=dummy_factory, windows=4)
    assert len(res["windows"]) == 4
    for w in res["windows"]:
        assert w["train_max_date"] < w["test_min_date"], "test window must be strictly after train window"
    assert res["pooled"]["log_loss"] < res["pooled"]["coinflip_log_loss"]
```

Run: `pytest tests/test_walk_forward_eval.py -v` — Expected: FAIL (`walk_forward_metrics` undefined).
- [ ] **Step 2: Implement** — expanding windows (e.g. train on seasons [t-k..t], test on season t+1, step forward); pooled metrics via concatenated out-of-fold predictions. Log-loss via `sklearn.metrics.log_loss`, Brier via `brier_score_loss`, AUC via `roc_auc_score`.
- [ ] **Step 3: Run the test** — Expected: PASS.
- [ ] **Step 4: Run against the real win model** — record pooled log-loss/Brier/AUC vs naive in a scratch note (feeds Task 11's report).
- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/models/evaluate/walk_forward_eval.py src/nba_predictor/models/evaluate/walk_forward.py tests/test_walk_forward_eval.py
git commit -m "feat: expanding-window walk-forward evaluation for win model"
```

### Task 4: Walk-forward for margin/total + naive scale + manifest emission

**Files:**
- Modify: `src/nba_predictor/models/evaluate/walk_forward_eval.py` (add margin/total path), `models/manifest.json` writer (wherever the manifest is written — check `models/` writer or `pipeline/`)
- Test: extend `tests/test_walk_forward_eval.py`

**Interfaces:**
- Produces: `walk_forward_regression(df, model_factory, target) -> dict` with `mae` per window + pooled, `naive_mae` (league-average margin/total baseline and a "home −3 / league-avg total" baseline); manifest gains `log_loss`, `brier`, `auc`, `margin_mae`, `total_mae`.

- [ ] **Step 1: Write the failing test**

```python
def test_margin_walk_forward_reports_mae_vs_naive_scale():
    df = make_games(n=100)  # with home_margin label
    res = walk_forward_regression(df, model_factory=dummy_reg_factory, target="home_margin")
    assert "mae" in res["pooled"] and "naive_mae" in res["pooled"]
    assert res["pooled"]["mae"] > 0

def test_manifest_emits_probability_metrics():
    manifest = build_manifest(metrics={"log_loss": 0.63, "brier": 0.22, "auc": 0.60})
    assert manifest["log_loss"] == 0.63 and manifest["brier"] == 0.22 and manifest["auc"] == 0.60
```

Run: `pytest tests/test_walk_forward_eval.py -v` — Expected: FAIL.
- [ ] **Step 2: Implement** — regression walk-forward mirroring Task 3; wire the three probability metrics + two MAEs into the manifest writer (this lights up the frontend `METRIC_NAMES` labels that are currently dead).
- [ ] **Step 3: Run the test** — Expected: PASS.
- [ ] **Step 4: Record** the real margin/total MAE vs naive scale (current: 13.9/18.6 — report honestly whatever the walk-forward says).
- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/models/evaluate/walk_forward_eval.py tests/test_walk_forward_eval.py
git commit -m "feat: walk-forward for margin/total vs naive scale; emit prob metrics to manifest"
```

### Task 5: Candidate race (Ridge vs XGBoost) + manifest winner

**Files:**
- Create: `src/nba_predictor/models/candidate_race.py`
- Modify: `src/nba_predictor/models/game_outcome.py` (serve the winner; keep loser code), manifest writer (`chosen_candidate`)
- Test: `tests/test_candidate_race.py`

**Interfaces:**
- Consumes: `walk_forward_metrics` / `walk_forward_regression` from Task 3–4.
- Produces: `run_candidate_race(df) -> {"winner": "ridge"|"xgboost", "results": {...}}`; manifest records `chosen_candidate`.

- [ ] **Step 1: Write the failing test**

```python
def test_race_picks_lower_walk_forward_logloss():
    df = make_games(n=120)
    res = run_candidate_race(df, candidates={"ridge": ridge_factory, "xgboost": xgb_factory})
    assert res["winner"] in ("ridge", "xgboost")
    # winner must have the min pooled log-loss
    losses = {k: v["pooled"]["log_loss"] for k, v in res["results"].items()}
    assert losses[res["winner"]] == min(losses.values())
```

Run: `pytest tests/test_candidate_race.py -v` — Expected: FAIL.
- [ ] **Step 2: Implement** — candidates: Elo baseline (log-loss reference only, never served), Ridge classifier/regressor, XGBoost with current hyperparams (`n_estimators=200, max_depth=4`). Do not assume XGBoost wins — NFL's Ridge beat it. Keep the race deterministic (fixed seeds).
- [ ] **Step 3: Run the test** — Expected: PASS.
- [ ] **Step 4: Run the real race** on the training data; record the winner and margins.
- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/models/candidate_race.py src/nba_predictor/models/game_outcome.py tests/test_candidate_race.py
git commit -m "feat: candidate race (Ridge vs XGBoost) with manifest winner"
```

### Task 6: Residual-σ probability math (coherent win/cover/over)

**Files:**
- Create: `src/nba_predictor/models/probability.py`
- Modify: `src/nba_predictor/models/game_outcome.py` (use it for serving), manifest writer (store `margin_sigma`, `total_sigma`)
- Test: `tests/test_probability.py`

**Interfaces:**
- Consumes: predicted margin `m`, predicted total `t`, residual σs from walk-forward residuals.
- Produces: `win_prob(m, sigma_m)`, `cover_prob(m, line, sigma_m)`, `over_prob(t, line, sigma_t)` via Normal CDF; `cover_prob` returns `None` when no real line exists (never a placeholder).

- [ ] **Step 1: Write the failing test**

```python
def test_cover_prob_none_without_real_line():
    assert cover_prob(predicted_margin=5.0, line=None, sigma=13.0) is None

def test_probs_are_coherent():
    # larger predicted margin -> higher win prob, monotone
    assert win_prob(10.0, 13.0) > win_prob(2.0, 13.0) > win_prob(-5.0, 13.0)
    # symmetry: P(home covers -x) == P(away covers +x) pattern holds via CDF
    assert abs(cover_prob(5.0, -5.0, 13.0) - 0.5) < 0.02
```

Run: `pytest tests/test_probability.py -v` — Expected: FAIL.
- [ ] **Step 2: Implement** with `scipy.stats.norm.cdf`. Fit σs from walk-forward residuals (pooled out-of-fold errors), store in manifest.
- [ ] **Step 3: Run the test** — Expected: PASS.
- [ ] **Step 4: Wire into serving** — `game_outcome.py` serves win/cover/over from this module; raw `predict_proba` no longer served directly.
- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/models/probability.py src/nba_predictor/models/game_outcome.py tests/test_probability.py
git commit -m "feat: residual-sigma Normal-CDF probability math; cover only on real lines"
```

### Task 7: Game features (pace, opponent-adjusted, rest, injury deltas)

**Files:**
- Create: `src/nba_predictor/features/game_context.py`
- Test: `tests/test_features_game_context.py`

**Interfaces:**
- Consumes: game frame + team game logs; ESPN availability feed shape (the same data `api/availability.py` uses).
- Produces: `build_game_features(games, team_logs, availability) -> DataFrame` with `pace`, `opp_adj_net`, `rest_delta`, `injury_delta`; every column shift(1)-then-rolling.

- [ ] **Step 1: Write the failing tests**

```python
def test_features_are_causal_shift1():
    feats = build_game_features(games, logs, availability)
    # A game-G feature may only use games < G: spot-check by recomputing one
    # team's rolling net rating excluding game G and comparing.
    assert causal_check(feats, games)

def test_no_all_nan_feature_columns():
    feats = build_game_features(games, logs, availability)
    for col in ["pace", "opp_adj_net", "rest_delta", "injury_delta"]:
        assert feats[col].notna().mean() > 0.8, f"{col} is mostly NaN — broken pull, not no-signal"

def test_injury_delta_uses_only_verified_feed():
    # availability rows flagged 'estimated' or from any non-ESPN source must not move the delta
    feats = build_game_features(games, logs, availability_with_unverified_row)
    assert (feats["injury_delta"] == feats_baseline["injury_delta"]).all()
```

Run: `pytest tests/test_features_game_context.py -v` — Expected: FAIL.
- [ ] **Step 2: Implement** — pace (possessions/48), opponent-adjusted net rating (iterative or simple SOS adjustment), rest-days delta (home rest − away rest), injury delta (net rating impact of Out/Doubtful players vs their season average, ESPN feed only).
- [ ] **Step 3: Run the tests** — Expected: PASS.
- [ ] **Step 4: Commit**

```bash
git add src/nba_predictor/features/game_context.py tests/test_features_game_context.py
git commit -m "feat: game-context features (pace, opp-adjusted, rest, injury deltas)"
```

### Task 8: Props chronological holdout

**Files:**
- Create: `src/nba_predictor/models/prop_holdout.py`
- Test: `tests/test_prop_holdout.py`

**Interfaces:**
- Consumes: player-game frame with `game_date`, the 4 target columns (points/rebounds/assists/threes), existing prop model factories.
- Produces: `prop_holdout_metrics(df, factories) -> {market: {"mae": ..., "in_sample_mae": ...}}` — the honest comparison the repo has never had.

- [ ] **Step 1: Write the failing test**

```python
def test_holdout_is_chronological_and_reports_both():
    df = make_player_games(n=200)
    res = prop_holdout_metrics(df, factories=dummy_prop_factories)
    for market in ("points", "rebounds", "assists", "threes"):
        assert res[market]["holdout_mae"] >= 0
        assert "in_sample_mae" in res[market]
        # holdout window strictly after train window
        assert res[market]["train_max_date"] < res[market]["holdout_min_date"]
```

Run: `pytest tests/test_prop_holdout.py -v` — Expected: FAIL.
- [ ] **Step 2: Implement** — chronological split (last 20% of player-games by date), train on first 80%, report both MAEs. This will likely show holdout MAE ≥ in-sample — report honestly.
- [ ] **Step 3: Run the test** — Expected: PASS.
- [ ] **Step 4: Commit**

```bash
git add src/nba_predictor/models/prop_holdout.py tests/test_prop_holdout.py
git commit -m "feat: chronological holdout evaluation for prop models"
```

### Task 9: Prop features (opponent-D-vs-position, rest, usage, position-keyed)

**Files:**
- Create: `src/nba_predictor/features/prop_matchup.py`
- Modify: `src/nba_predictor/features/player_stats.py` (extend, don't break the 5 rolling means)
- Test: `tests/test_features_prop_matchup.py`

**Interfaces:**
- Consumes: player-game logs, team defensive ratings by position, schedule (rest).
- Produces: `build_prop_features(...)` adding `opp_def_vs_pos`, `rest_days`, `usage_trend`, `minutes_trend`; position groups G/F/C.

- [ ] **Step 1: Write the failing tests** — mirror Task 7's three tests (causal shift(1), no all-NaN columns with >0.8 coverage, position groups present as G/F/C).
- [ ] **Step 2: Implement** — opponent defensive rating vs the player's position group (trailing 10 games, shift(1)); rest days since last game; usage% and minutes trailing trends.
- [ ] **Step 3: Run holdout (Task 8) with new features** — record MAE delta vs the 5-rolling-means baseline.
- [ ] **Step 4: Commit**

```bash
git add src/nba_predictor/features/prop_matchup.py src/nba_predictor/features/player_stats.py tests/test_features_prop_matchup.py
git commit -m "feat: prop matchup features (opp-D-vs-position, rest, usage, position-keyed)"
```

### Task 10: Offline artifact gate + per-player ledger writers/readers

**Files:**
- Create: `src/nba_predictor/models/artifact_gate.py`
- Modify: `src/nba_predictor/tracking/store.py` (writers/readers for `game_player_outcomes` + `player_prediction_snapshots` — tables exist, L23–87)
- Test: `tests/test_artifact_gate.py`, `tests/test_prop_ledger.py`

**Interfaces:**
- Consumes: holdout metrics from Task 8.
- Produces: `gate_artifact(metrics) -> bool` — refuses to write the pickle if calibration fails (±5pt/bucket at n≥100, NFL pattern); `write_prop_snapshot(...)`, `read_prop_ledger(...)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_gate_refuses_uncalibrated_model():
    assert gate_artifact({"calibration_max_gap": 0.09, "n": 500}) is False
    assert gate_artifact({"calibration_max_gap": 0.03, "n": 500}) is True
    assert gate_artifact({"calibration_max_gap": 0.01, "n": 40}) is False  # n too small

def test_prop_snapshot_roundtrip_with_position():
    write_prop_snapshot(game_id="g1", player="P1", position="G", market="points", pred=23.5, ts=PRE_TIP)
    rows = read_prop_ledger(game_id="g1")
    assert rows[0]["position"] == "G" and rows[0]["pred"] == 23.5

def test_player_id_join_never_guesses():
    # unmatched (name, team) pairs are logged + skipped, never fuzzy-matched
    assert join_player_ids([("Zzz Unknown", "XXX")], espn_players) == []
```

Run both — Expected: FAIL.
- [ ] **Step 2: Implement** the gate and the ledger writers/readers (normalized (name, team) join; log-and-skip unmatched).
- [ ] **Step 3: Run the tests** — Expected: PASS.
- [ ] **Step 4: Commit**

```bash
git add src/nba_predictor/models/artifact_gate.py src/nba_predictor/tracking/store.py tests/test_artifact_gate.py tests/test_prop_ledger.py
git commit -m "feat: prop artifact gate + per-player ledger writers/readers"
```

### Task 11: Honest evaluation report + Phase A review gate

**Files:**
- Create: `docs/nba-parity-evaluation-2026-10.md`

- [ ] **Step 1: Run everything** — walk-forward (Tasks 3–4), candidate race (Task 5), prop holdout with new features (Tasks 8–9). Collect: win log-loss/Brier/AUC vs naive; margin/total MAE vs naive scale; race winner + margins; prop holdout MAE per market and per position vs in-sample.
- [ ] **Step 2: Write the report** — one table per model, metric vs naive, honest commentary. If a model doesn't beat naive, say so in the report — that is a successful task outcome, not a failure.
- [ ] **Step 3: Open the Phase A PR** against `main` (branch `nba-parity-review`). PR description links the spec and the evaluation report.
- [ ] **Step 4: Commit the report**

```bash
git add docs/nba-parity-evaluation-2026-10.md
git commit -m "docs: honest Phase A evaluation report"
git push origin nba-parity-review
```

**STOP — Kevin reviews the PR and the evaluation report before Phase B starts.**

---

## Phase B — Value layer (P1)

### Task 12: Edge gate (5%, one single/game, no parlays, ≤1h freshness, same-book pairing)

**Files:**
- Create: `src/nba_predictor/odds/edge_gate.py`
- Modify: `src/nba_predictor/api/routes.py` (value-pick endpoint/section)
- Test: `tests/test_edge_gate.py`

**Interfaces:**
- Consumes: the odds payload the refresh loop already produces (PR #32 — do not reschedule anything).
- Produces: `gated_picks(odds_rows, model_probs) -> list` — only rows passing all gates.

- [ ] **Step 1: Write the failing tests**

```python
def test_gate_applies_all_rules():
    rows = [
        fresh_row(edge=0.06, game="g1"),   # passes
        fresh_row(edge=0.04, game="g1"),   # fails 5% threshold
        stale_row(edge=0.09, game="g2"),   # fails 1h freshness
        fresh_row(edge=0.07, game="g1", second_best=True),  # fails one-single-per-game
    ]
    picks = gated_picks(rows, model_probs)
    assert [p["id"] for p in picks] == ["fresh_row_6pct"]

def test_totals_pair_same_book_same_point():
    # a total edge computed across different books/points is rejected
    assert pair_total(home_rows_mixed_books) is None
```

Run: `pytest tests/test_edge_gate.py -v` — Expected: FAIL.
- [ ] **Step 2: Implement** — 5% edge threshold, max one single per game (highest edge wins), no parlays, odds timestamp ≤1h old (else zero flags), totals paired same-book/same-point (NFL's post-review fix).
- [ ] **Step 3: Run the tests** — Expected: PASS.
- [ ] **Step 4: Expose** the gated picks in the API (new endpoint or section of the odds payload — match the existing API style).
- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/odds/edge_gate.py src/nba_predictor/api/routes.py tests/test_edge_gate.py
git commit -m "feat: 5% edge gate (one single/game, no parlays, 1h freshness)"
```

### Task 13: Graded value ledger + CLV + weekly report

**Files:**
- Create: `src/nba_predictor/tracking/value_ledger.py`
- Modify: `src/nba_predictor/tracking/store.py` (ledger table, ALTER TABLE ADD COLUMN pattern)
- Test: `tests/test_value_ledger.py`

**Interfaces:**
- Consumes: gated picks from Task 12.
- Produces: `snapshot_pick(...)` (immutable, pre-tip), `settle_pick(...)` (fills outcome), `record_closing_line(...)` → CLV per pick; `weekly_ledger_report()` markdown.

- [ ] **Step 1: Write the failing tests** — snapshot immutability (second snapshot of same game/market doesn't displace the first), post-tip snapshot rejected, CLV computed vs recorded closing line, weekly report renders picks/hit-rate/yield/CLV.
- [ ] **Step 2: Implement** — table + snapshot/settle/close functions; honest copy in the report header ("not evidence of a profitable strategy" — PL's framing).
- [ ] **Step 3: Run the tests** — Expected: PASS.
- [ ] **Step 4: Commit**

```bash
git add src/nba_predictor/tracking/value_ledger.py src/nba_predictor/tracking/store.py tests/test_value_ledger.py
git commit -m "feat: graded value ledger with CLV + weekly report"
```

### Task 14: Snapshot staleness gate

**Files:**
- Modify: `src/nba_predictor/public_snapshot.py`
- Test: `tests/test_snapshot_staleness.py` (check existing snapshot tests first — extend, don't duplicate)

- [ ] **Step 1: Write the failing test** — publishing with a 20-day-old model is refused; with a 2-day-old model it succeeds. Suggested max age: 14 days (Kevin decides the constant — make it a named constant).
- [ ] **Step 2: Implement** — port NFL `public_snapshot.py` L226–272: compare model `trained_at` vs now at publish time; refuse with a logged error.
- [ ] **Step 3: Run the tests** — Expected: PASS.
- [ ] **Step 4: Commit**

```bash
git add src/nba_predictor/public_snapshot.py tests/test_snapshot_staleness.py
git commit -m "feat: refuse snapshot publish when models are stale"
```

### Task 15: Dead-code cleanup

**Files:**
- Delete: `src/nba_predictor/data/odds_api.py`, `src/nba_predictor/data/balldontlie.py`, `src/nba_predictor/data/nba_api.py`, their dedicated tests if they only cover the dead modules.
- Modify: `src/nba_predictor/tracking/store.py` — drop `odds_timing_snapshots` / `game_forecast_snapshots` (no writer/reader) or wire them; `get_player_props` — validate against the live sportsbook schema or delete.

- [ ] **Step 1: Verify deadness** — `grep -rn "odds_api\|balldontlie\|nba_api" src/ tests/ --include="*.py" | grep -v test_data_odds_api | grep -v test_data_balldontlie | grep -v test_data_nba_api` must show no production imports before deleting. If something imports them, it's not dead — wire it or report back.
- [ ] **Step 2: Delete + fix imports.**
- [ ] **Step 3: Full suite** — `python -m pytest tests/ -x -q` must pass.
- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove dead odds/data modules and unused tracking tables"
```

**STOP — Phase B PR. Kevin reviews the ledger's first weekly report before Phase C.**

---

## Phase C — Polish, preseason, ops truth (P2)

**Frontend method for every task below:** impeccable (spec §16). Build with
`/impeccable craft` or `polish`, then `/impeccable audit` each touched surface
and fix detector findings before committing. Ponytail still applies — the
laziest diff that passes the audit.

### Task 16: Game cards — status, timing badges, sort

**Files:** frontend game-list components (find the card component — check `frontend/src/components/`).

- [ ] Add game status (FINAL + HIT/MISSED, LIVE, NEXT UP with tip time) and pick-timing badges on the cards themselves (detail dialog already has "MADE BEFORE TIP-OFF").
- [ ] Add sort options: kickoff order / most confident first (NFL pattern).
- [ ] Acceptance: cards for a finished, live, and upcoming game render the right status; sort toggles reorder; no console errors.

### Task 17: Game detail — model-vs-line, uncertainty, injury line

- [ ] Add an "Other model markets" block: spread cover chance, over/under chance, projected margin with ±, model-vs-line with edge (the market odds table already has the data — surface the comparison).
- [ ] Add the injury-report line ("Checked against ESPN availability — X Out, Y Day-to-Day" or the gate's actual state).
- [ ] Acceptance: detail dialog shows the block for a game with a real line and hides cover chance when no real line exists (Task 6's `None` path).

### Task 18: Track record — per-pick table, upsets, projected standings

- [ ] Sortable per-pick detail table (game, market, pick, actual, hit/miss, when made) — NFL pattern.
- [ ] Biggest-upsets/misses table — PL pattern.
- [ ] Projected final standings panel — PL's ProjectedTable pattern.
- [ ] Acceptance: tables render from the tracking API; sorting works; empty states are honest.

### Task 19: Timezone label + dead labels

- [ ] Label tip times (NFL: "Kickoff times in UTC" — use the site's actual convention).
- [ ] Verify `METRIC_NAMES` — Task 4 made the backend emit log_loss/brier/AUC; remove any labels that are still dead.
- [ ] Acceptance: no dead labels remain; times carry a timezone.

### Task 20: Preseason flag

**Files:** `src/nba_predictor/ingest.py` (L422–467 playoff_status), API + frontend.

- [ ] Add a season-state flag (offseason/preseason/regular/postseason) derived from the schedule.
- [ ] Suppress or relabel `playoff_status` and projections during offseason (no more "Clinched"/"eliminated" from empty records); Data Hub keeps its honest "off-season · last season's numbers" labeling.
- [ ] Test: offseason schedule → `playoff_status` is `None`/suppressed, never "Clinched".
- [ ] Commit with the test.

### Task 21: Ops truth

- [ ] `deploy-azure.yml` is manual-only ("Everything runs on the VPS now") while README claims every push redeploys. Either restore push-triggered VPS deploy (NFL/PL `deploy.yml` → GHCR → `deploy <sport> <sha>` with health-check + rollback) or fix the README. **Kevin decides which** — ask in the Phase C PR if unsure; default to fixing the README (smallest honest change).
- [ ] Acceptance: README and workflow agree; no stale auto-deploy claim.

**Phase C PR. After Kevin's approval, merge to `main`.**

---

## Self-review

**1. Spec coverage:** Every open gap in spec §2.3 has a task: G1→Tasks 3–4, G2→Tasks 5–6, G3→Tasks 1–2, G4→Tasks 8–10, G5→Tasks 12–13, G6→Task 14, G7→Task 15, G8→Task 7, G9→Task 20, G10→Tasks 4+19, G11→Tasks 16–18, G12→Task 21. The "already done" items in spec §2.1 have no tasks (verified in the working tree, not redone). PR #32's odds loop is consumed by Task 12, not rebuilt.

**2. Placeholder scan:** No TBD/TODO/"similar to Task N" — each task names exact files, interfaces, test code, and commit commands. Phase C tasks are coarser (frontend acceptance criteria instead of 5-step TDD) because they're UI work against an existing component tree; each still has concrete acceptance checks.

**3. Type consistency:** `walk_forward_metrics`/`walk_forward_regression` signatures are defined in Tasks 3–4 and consumed as such in Task 5. `gated_picks`, `snapshot_pick`/`settle_pick`, `gate_artifact`, `build_game_features`, `build_prop_features` are each defined once at first use. The manifest gains `chosen_candidate`, `margin_sigma`, `total_sigma`, `log_loss`, `brier`, `auc` — Task 4 emits the metrics, Tasks 5–6 add the rest; no task assumes a field an earlier task didn't create.

**4. Review Focus:** All five failure modes have pinning tests: leakage → Task 7 shift(1) tests; frozen data → Task 2 never-frozen test; post-tip snapshots → Task 12 freshness tests; player-ID mismatch → Task 10 join test; all-NaN columns → Task 7 coverage test.
