# NBA Predictor — Design Spec

Date: 2026-09-15
Status: Approved for planning

## 1. Summary

A new sibling project to PL_Predictor, NFL_Predictor, and CFB_Predictor: an NBA
game-outcome and player-prop prediction system with a PL_Predictor-level
frontend (Fixtures page, tabbed Data Hub, Player Hub, model Track Record,
projected final standings). It follows the same architecture, deploy target,
and data-source patterns as its siblings so it can be built, hosted, and
operated the same way, and eventually listed on the shared `predictor-hub`
landing page (a placeholder "In development" card has already been added
there).

## 2. Stack & hosting

Mirrors PL_Predictor's stack (richest of the three siblings) and all four
siblings' deploy pattern:

- **Backend**: Python 3.13, FastAPI + Uvicorn, package `nba_predictor` under
  `src/`.
- **ML/data libs**: pandas, numpy, scipy, `xgboost`, `scikit-learn`, `optuna`
  (hyperparameter tuning), `nba_api`, `requests`.
- **Frontend**: React 19 + TypeScript, Vite, **Tailwind CSS v4**
  (`@tailwindcss/vite`), Recharts for charts. No component library — hand-built
  Tailwind components, matching PL_Predictor.
- **Database**: SQLite `data/tracking.db` for prediction history/tracking
  (`predictions`, `fixture_market_predictions`, `fixture_forecast_snapshots`,
  `odds_timing_snapshots`, `player_prediction_snapshots`,
  `fixture_player_outcomes` tables — same schema shape as PL_Predictor's
  tracking store, renamed for games/players as appropriate). Everything else
  (raw source data) is cached JSON/CSV on disk under `data/cache/<source>/`.
- **Deploy**: Docker multi-stage build (Node 20 build stage → python:3.13-slim
  runtime), pushed to GHCR, deployed to a new Azure Container App
  `nba-predictor` in the existing `predictor-hub-rg` resource group —
  `.github/workflows/deploy-azure.yml` copied and adapted from PL_Predictor.
  Because Azure Container Apps' filesystem isn't reliable for SQLite over
  network storage, adopt NFL/CFB_Predictor's workaround: live `tracking.db` on
  ephemeral container disk, backed up to/restored from a persistent volume
  path (`TRACKING_DB_BACKUP_PATH`) around each tracking tick.
- **Public snapshot**: `data/public_snapshot.json`, generated via
  `python -m nba_predictor.public_snapshot`, committed to git, refreshed on a
  schedule by `.github/workflows/refresh-public-snapshot.yml`, and polled
  every `PUBLIC_SNAPSHOT_POLL_SECONDS` (300s) by the running instance so a
  public deployment stays current without a redeploy. `PUBLIC_MODE` flag
  gates admin/write routes (404s them) on the public deployment; reuse
  PL_Predictor's `GuestAuthMiddleware` pattern if a password gate is wanted.
- **predictor-hub integration**: out of scope for this build. A placeholder
  "In development" card was already added to
  `/Users/sigey/Documents/Projects/predictor-hub/index.html`. Once
  NBA_Predictor is deployed and live, update that card to a live link (a
  separate, later task).

## 3. Data pipeline

| Source | Role | Auth | Notes |
|---|---|---|---|
| **nba_api** | Primary stats engine | None (unofficial stats.nba.com wrapper) | Play-by-play, box scores, Four Factors, hustle stats, tracking/synergy play-types, lineup on/off data. Rate-limit-sensitive and no formal ToS guarantee — cache aggressively, add retry/backoff. |
| **balldontlie** | Secondary/fallback stats & schedule | API key (existing) | Official REST API, 5 req/min free tier. Used for schedule confirmation and as a box-score fallback if nba_api is rate-limited. |
| **RapidAPI Sportsbook API** | Primary odds source, incl. player props | Existing `SPORTSBOOK_API_KEY` (RapidAPI account already covers NBA) | Same client pattern as PL_Predictor's `sportsbook_api.py`: per-event fetch, ~150 req/day cap, cache ~6h. |
| **The Odds API** | Odds fallback | Existing `ODDS_API_KEY` | Sport key `basketball_nba`. Bulk fetch `h2h,spreads,totals`. Used when RapidAPI is unavailable/exhausted. |
| **ESPN unofficial API** | Near-tipoff injury/lineup status | None | Same pattern as PL/NFL's `espn.py`. |
| **nbainjuries / NBA official injury report** | Injury status feed for feature engineering | None | Historical + current injury status, used to compute a missing-player production-value feature. |
| **Static team reference table** | Arena lat/long, timezone, altitude, conference/division | N/A (hardcoded, 30 rows) | Needed for travel-mileage, timezone-change, and altitude features — no API required. |

Cache directories: `data/cache/{nba_api, balldontlie, odds, sportsbook,
injuries, espn}/`. Refresh cadence and manual-refresh admin endpoints follow
the PL_Predictor pattern (`POST /refresh-odds`, `POST /refresh-fixtures`
equivalents).

## 4. Model & features

**Approach**: XGBoost-based, consistent with current NBA prediction research
(XGBoost/ensemble approaches outperform simpler baselines; defensive
rebounding, eFG%, TOV%, and FT% are consistently the strongest drivers of
outcome).

- **Game outcome**: XGBoost classifier for win probability, plus XGBoost
  regressors for predicted spread (margin) and total points.
- **Player props**: XGBoost regressors for points, rebounds, assists,
  three-pointers made; XGBoost classifier for double-double probability.
  Modeled the same way NFL_Predictor maps one model per stat/position.

**Feature set** (`src/nba_predictor/features/build.py`, rolling +
contextual, chronological train/validation split — never random, to avoid
temporal leakage):

- **Four Factors**, rolling for/against: effective FG%, turnover rate,
  offensive/defensive rebound rate, free-throw rate
- **Efficiency**: offensive rating, defensive rating, net rating, pace
- **Power rating**: Elo-style rating with home-court adjustment
- **Rest/fatigue**: rest days, back-to-back flag, 3-games-in-4-nights and
  4-games-in-6-nights congestion flags
- **Travel**: rolling 7-day mileage, timezone-change count, a composite
  fatigue index blending the two with rest deficit
- **Injuries**: missing-player production-value sum (season per-player stats
  weighted by recent minutes/usage)
- **Context**: head-to-head history, current win/loss streak, altitude flag
  (Denver), conference/division game flag

**Training/retraining**: `models/manifest.py`, walk-forward validation
(`evaluate/walk_forward.py`), calibration (`evaluate/calibration.py`) — same
shape as siblings. Artifacts (`game_outcome_model.pkl`,
`spread_model.pkl`, `total_points_model.pkl`, `points_model.pkl`,
`rebounds_model.pkl`, `assists_model.pkl`, `threes_model.pkl`,
`double_double_model.pkl`, `manifest.json`) committed to `models/` so the
Docker image ships with a working model. Retrain triggered via admin-only
`POST /retrain`.

## 5. Markets covered

Moneyline (h2h), spread, total (over/under), and player props (points,
rebounds, assists, threes made, double-double) — full "value bet" treatment
(model probability vs. market-implied probability, edge %, highlighted in the
UI) for every market with a live sportsbook line, matching PL_Predictor's
approach. De-vig market odds using the same Shin two-way model NFL_Predictor
uses (`odds/value_bets.py`) rather than naive implied-probability
normalization.

## 6. Frontend

Structure mirrors PL_Predictor's `frontend/src/pages/` richness:

- **Games (Fixtures) page**: game list for the current/selectable date or
  week; clicking a game opens a modal showing team form strips, rest/travel/
  injury context table, head-to-head history, win probability, predicted
  score & margin, spread/total distributions with value-bet highlighting
  (model % vs market %, edge %, bookmaker), and a player-prop highlights
  section (top scorer/rebounder/assist projections per team). Post-game:
  prediction review (verdict vs. actual) and player-call review, same as
  PL_Predictor's `FixtureModal`.
- **Data Hub** (tabbed page, auto-refresh every 60s, each tab independently
  error-tolerant):
  - **Team Hub**: teams grouped by conference/division, sortable team list +
    detail panel (Four Factors, pace, ratings trend, W-L record, current
    streak).
  - **Player Hub**: sortable/groupable (by team/position), paginated table —
    rating, live form rating, PPG/RPG/APG, shooting splits (FG%/3P%/FT%),
    usage rate, minutes.
  - **Power Rankings**: league-wide power rating ladder.
  - **Projected Standings**: full-season projection per conference —
    Eastern/Western standings, playoff seeding, play-in probabilities (this
    is the "predicted finishing positions" requirement).
  - **Track Record**: model hit-rate summary across markets.
- **Model Summary page** (public-safe): headline manifest metrics, model
  choice, feature importance chart.
- **Calibration page** (private/admin): reliability curves, backtest panel,
  live value-bet panel, walk-forward betting panel, model freshness panel,
  retrain trigger button.
- **Design system**: Tailwind v4, dark-mode-first, Inter font (Google
  Fonts), radial-gradient background — same mechanics as PL_Predictor's
  theme but with its own color identity: a navy/orange tonal ramp (`--color-
  nba-950` … `--color-nba-500`) distinct from PL's purple and NFL's purple
  accent, avoiding any single real team's trademarked colors.

## 7. API routes

Mirrors PL_Predictor's route shape, renamed for games:

`/health`, `/games`, `/games/{date-or-week}`, `/games/{game_id}` (detail),
`/games/{game_id}/players`, `/games/{game_id}/player-review`, `/teams`,
`/teams/{team}/games`, `/manifest`, `/manifest/history`, `/calibration`,
`POST /backtest` (admin), `/value-bets/walk-forward`,
`/value-bets/track-record`, `POST /retrain` (admin), `POST /refresh-odds`
(admin) / `POST /refresh-odds/public`, `POST /refresh-games` (admin),
`/hub/rankings`, `/hub/standings`, `/hub/track-record`, `/hub/teams`,
`/hub/players`, `/prop-track-record`. `PUBLIC_MODE` gates admin/write
endpoints on the public deployment.

## 8. Repo skeleton

```
NBA_Predictor/
  Dockerfile
  pyproject.toml
  README.md
  data/
    cache/{nba_api,balldontlie,odds,sportsbook,injuries,espn}/
    tracking.db
    public_snapshot.json
  docs/
    superpowers/specs/
  frontend/
    src/
      pages/  (GamesPage, DataHubPage, ModelSummaryPage, CalibrationPage)
      components/
      index.css
  models/
    manifest.json
    *.pkl
  src/nba_predictor/
    api/routes.py
    data/  (nba_api.py, balldontlie.py, sportsbook_api.py, odds_api.py, espn.py, injuries.py, team_reference.py)
    features/build.py
    models/  (game_outcome.py, player_props.py, season_projection.py, manifest.py)
    odds/value_bets.py
    tracking/store.py
    public_snapshot.py
    config.py
  tests/
  .github/workflows/
    deploy-azure.yml
    refresh-public-snapshot.yml
```

## 9. Explicitly out of scope for this build

- Making `predictor-hub`'s NBA card live (follow-up once deployed).
- WNBA or any league beyond NBA.
- Live in-game / in-play predictions (F1_Predictor has live-session support;
  not requested here — pre-game predictions only, refreshed as odds/injuries
  update pre-tipoff).
