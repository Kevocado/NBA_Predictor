# Retrain stall — diagnosis (2026-10-05)

Task 1 of `docs/superpowers/plans/2026-10-04-nba-parity.md`. Closes gap **G3**.

## TL;DR

The daily retrain has **never succeeded once**. All 17 `refresh-data.yml` runs
available through `gh run list` (2026-09-18 → 2026-10-04) concluded `failure`;
there are zero successes. The workflow crashes *before* it writes anything, so
`models/manifest.json` and `models/manifest_history.jsonl` are still the ones a
human committed on 2026-09-18 — hence the byte-identical metrics in all four
history entries.

**The stall is a too-narrow training window, not a cache-path mismatch.**

## What the logs say

`gh run view 37215407314 --log-failed` (the 2026-10-04 run), in order:

```
Fetching schedule 2026-08-05 to 2026-10-18 from ESPN...
  66 games found
Fetching box scores for completed games...
  wrote .../data/cache/schedule/games.json
Fetching player box scores for 66 games since 2026-09-27...
  wrote hub caches to .../data/cache/hub
Training on 1 completed games with full box scores...
ValueError: Need at least 2 rows to perform a chronological split
```

The pull itself is healthy: the schedule fetch, the box-score fetch and the hub
cache writes all succeed. The window is the problem.

## Root cause

`.github/workflows/refresh-data.yml` computes the window as **rolling**:

```yaml
echo "start=$(date -u -d '60 days ago' +%Y-%m-%d)" >> "$GITHUB_OUTPUT"
echo "end=$(date -u -d '14 days' +%Y-%m-%d)" >> "$GITHUB_OUTPUT"
```

That 60-day-back window is fine for *scoring upcoming games* and wrong for
*training*. Two compounding failures:

1. **Offseason (the current failure).** On 2026-10-04 the window is
   2026-08-05 → 2026-10-18. The NBA regular season has not started, so of 66
   games ESPN returned, exactly **one** is completed *and* carries the full box
   score. `to_training_frame` (`pipeline/ingest.py:288`) keeps only rows with
   `completed` and `home_fgm` present, so `training_df` has one row.
   `run_retrain_pipeline` (`pipeline/retrain.py:21`) hands that to
   `chronological_split`, which raises
   `ValueError: Need at least 2 rows to perform a chronological split`.
   The `python -m nba_predictor.pipeline.ingest` step exits 1.
2. **The commit step is downstream of the crash.** "Regenerate public snapshot"
   and "Commit if changed" are separate steps, so the failure skips both. No
   manifest write, no `manifest_history.jsonl` append, no snapshot. The
   workflow is not "retraining on stale data" — it is not retraining at all.

Even in-season the 60-day window is the wrong shape: the rolling features in
`features/build.py` need a season of history behind them, and 60 days is roughly
one quarter of one.

### Where data stops flowing

| Layer | File:line | State |
|---|---|---|
| Window computed | `.github/workflows/refresh-data.yml:22-26` | **breaks here** — 60d back, cannot cover a season |
| Games filtered to trainable rows | `src/nba_predictor/pipeline/ingest.py:288-307` | correct, just starved |
| Split raises | `src/nba_predictor/models/evaluate/walk_forward.py:6` | crash, uncaught |
| Manifest written | `src/nba_predictor/pipeline/retrain.py:56-62` | never reached |

## Two secondary findings (real, but not what caused the stall)

**1. `POST /retrain` reads a file nothing writes.** `api/deps.py:57` resolves
`data/cache/training/games.json`, and `api/routes.py:608-614` 400s when it is
missing. `grep -rn "cache.*training\|training.*games.json" src/ tools/ .github/`
finds **no writer anywhere** — and `data/cache/*` is gitignored except
`schedule/` and `hub/`, so the file can never ship in a deploy either. The
endpoint is unrunnable in every environment, including the VPS. The success
path is only reachable because `tests/test_api_admin.py:36` injects the path via
`dependency_overrides` and writes the fixture itself. The real retrain path in
production is `/admin/refresh-full` → `run_ingest`, which does work.

**2. The committed schedule cache can never satisfy training.**
`to_schedule_cache` (`pipeline/ingest.py:268-285`) writes only
`game_id / game_date / home_team / away_team / completed / home_pts / away_pts /
tip_off` — it deliberately drops every box-score field. The committed
`data/cache/schedule/games.json` holds 1760 games (2025-10-02 → 2026-11-30) and
**zero** completed rows carrying `home_fgm`. So pointing retrain at the schedule
cache instead of the training cache would not help; the box fields are missing
at the source, not at the path.

Its date range does confirm the committed data came from the *manual* path, not
the workflow: `api/routes.py:664` hardcodes `run_ingest("2025-10-01",
"2026-11-30")`, and the committed cache spans exactly 2025-10-02 → 2026-11-30.

## The fix Task 2 implements

1. **Season-anchored window.** Replace the rolling `60 days ago` with a start
   that reaches back to at least the start of the previous NBA season. Reuse the
   season-year rule already in `pipeline/retrain.py:17` (`_nba_season_start_year`)
   rather than inventing a second one — the window becomes "Oct 1 of the
   season before the current one" → "today + 14d", which covers two seasons of
   completed games and still scores the upcoming slate.
2. **Write the training cache the admin endpoint already reads.** One line in
   `run_ingest`, so `data/cache/training/games.json` exists with box-score
   fields and `POST /retrain` stops being unrunnable. Additive, no schema change.
3. **Never-frozen regression test** (`tests/test_retrain_freshness.py`): two
   retrains on different training data must append two non-identical
   `manifest_history.jsonl` entries. This pins the failure mode itself rather
   than the window constant.

## Not doing (deliberate)

- Not adding a fallback that trains on <2 games to make CI green. Skipping the
  retrain with a logged reason is honest; a model fitted on one game is not.
- Not widening `to_schedule_cache` to carry box scores. It is the *serving*
  cache and the snapshot contract; training data gets its own file (item 2).
- Not touching the rolling features. `shift(1)-then-rolling` is correct; it just
  needs enough history behind it, which item 1 supplies.