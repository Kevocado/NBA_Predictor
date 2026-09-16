# Phase 3: Model & Features Implementation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement XGBoost-based prediction models, feature engineering pipeline, and value bet analysis for NBA game outcomes and player props.

**Architecture:** Features are computed from cached data sources (nba_api, balldontlie, injuries). Models are XGBoost classifiers/regressors trained on historical data. Value bets compare model probability vs market-implied probability using the Shin two-way model.

**Tech Stack:** Python 3.13, XGBoost, scikit-learn, pandas, numpy, scipy, optuna, tenacity

**Spec:** `docs/superpowers/specs/2026-09-15-nba-predictor-design.md`

---

## Global Constraints

- Python version floor: **3.13**
- All data modules read API keys through `config.py`
- XGBoost models serialized as `.pkl` files in `models/`
- Feature engineering uses rolling windows (no random splits — temporal only)
- Cache files use JSON format under `data/cache/<source>/`
- Retry policy: exponential backoff, max 5 retries (via `tenacity`)
- Every module must have comprehensive tests in `tests/`

---

## File Structure

```
src/nba_predictor/
  features/
    __init__.py
    build.py           # Feature engineering pipeline
  models/
    __init__.py
    game_outcome.py    # XGBoost classifier for win probability
    spread.py          # XGBoost regressor for predicted spread
    player_props.py    # XGBoost regressors for player stats
    manifest.py        # Model manifest and metadata
  odds/
    value_bets.py      # Shin two-way model, value bet detection
  api/
    routes.py          # Additional routes: /games, /teams, /manifest, /hub/*
```

---

### Task 1: Features module

**Files:**
- Create: `src/nba_predictor/features/build.py`
- Test: `tests/test_features.py`

**Interfaces:**
- `build_features(game_data: dict, team_data: dict, injury_data: dict) -> dict`
  - Returns feature vector with: four_factors, efficiency, power_rating, rest_fatigue, travel, injuries, context
- `compute_four_factors(game: dict) -> dict`
- `compute_rest_fatigue(team_id: int, games: list[dict]) -> dict`
- `compute_travel(team_id: int, games: list[dict]) -> dict`

**Features to compute:**
- Four Factors (rolling for/against): effective FG%, turnover rate, offensive/defensive rebound rate, free-throw rate
- Efficiency: offensive rating, defensive rating, net rating, pace
- Power rating: Elo-style with home-court adjustment
- Rest/fatigue: rest days, back-to-back flag, congestion flags
- Travel: rolling 7-day mileage, timezone-change count, composite fatigue index
- Injuries: missing-player production-value sum
- Context: head-to-head history, win/loss streak, altitude flag, conference flag

- [ ] **Step 1: Write tests for build_features**
- [ ] **Step 2: Implement build_features with all feature categories**
- [ ] **Step 3: Implement helper functions (compute_four_factors, compute_rest_fatigue, compute_travel)**
- [ ] **Step 4: Run tests and verify all pass**
- [ ] **Step 5: Commit**

---

### Task 2: Model modules

**Files:**
- Create: `src/nba_predictor/models/game_outcome.py`
- Create: `src/nba_predictor/models/spread.py`
- Create: `src/nba_predictor/models/player_props.py`
- Create: `src/nba_predictor/models/manifest.py`
- Test: `tests/test_models.py`

**Interfaces:**
- `game_outcome.GameOutcomeModel.predict(features: dict) -> dict`
  - Returns `{"win_probability": float, "predicted_score": dict}`
- `spread.SpreadModel.predict(features: dict) -> dict`
  - Returns `{"predicted_spread": float, "confidence": float}`
- `player_props.PlayerPropsModel.predict(player_id: int, game_id: str) -> dict`
  - Returns `{"points": float, "rebounds": float, "assists": float, "threes": float, "double_double_prob": float}`
- `manifest.create_manifest() -> dict`
  - Returns model metadata with version, training date, feature importance

**Model types:**
- `XGBClassifier` for game outcome (win probability) and double-double
- `XGBRegressor` for spread, total points, player stats
- Rolling walk-forward validation for training
- Model artifacts saved as `.pkl` in `models/`

- [ ] **Step 1: Write tests for game_outcome model**
- [ ] **Step 2: Implement GameOutcomeModel**
- [ ] **Step 3: Write tests for spread model**
- [ ] **Step 4: Implement SpreadModel**
- [ ] **Step 5: Write tests for player_props model**
- [ ] **Step 6: Implement PlayerPropsModel**
- [ ] **Step 7: Implement manifest module**
- [ ] **Step 8: Run tests and verify all pass**
- [ ] **Step 9: Commit**

---

### Task 3: Value bets analysis

**Files:**
- Create: `src/nba_predictor/odds/value_bets.py`
- Test: `tests/test_value_bets.py`

**Interfaces:**
- `value_bets.shin_two_way(home_prob: float, away_prob: float) -> dict`
  - Returns `{"home_implied": float, "away_implied": float, "vig": float}`
- `value_bets.detect_value_bets(model_prob: float, market_prob: float, edge_threshold: float = 0.05) -> dict`
  - Returns `{"has_edge": bool, "edge_pct": float, "recommendation": str}`
- `value_bets.compute_value_bets(games: list[dict], market_odds: list[dict]) -> list[dict]`
  - Returns list of value bet opportunities with edge percentages

**Methods:**
- Shin two-way model for de-vigging market odds
- Edge calculation: model_probability vs market_implied_probability
- Value bet threshold: edge > 5% by default

- [ ] **Step 1: Write tests for value_bets**
- [ ] **Step 2: Implement shin_two_way**
- [ ] **Step 3: Implement detect_value_bets**
- [ ] **Step 4: Implement compute_value_bets**
- [ ] **Step 5: Run tests and verify all pass**
- [ ] **Step 6: Commit**

---

### Task 4: Extended API routes

**Files:**
- Modify: `src/nba_predictor/api/routes.py`
- Test: `tests/test_api_routes.py`

**New routes:**
- `GET /games` — list games for current/selectable date
- `GET /games/{game_id}` — game detail with prediction
- `GET /games/{game_id}/players` — player projections
- `GET /teams` — list all 30 teams
- `GET /teams/{team}/games` — team game history
- `GET /manifest` — model manifest
- `GET /hub/rankings` — power rankings
- `GET /hub/standings` — projected standings
- `GET /value-bets/walk-forward` — value bet summary

**Each route:**
- Uses data modules from Phase 2 for data
- Uses models from Task 2 for predictions
- Uses value_bets from Task 3 for edge detection
- Returns JSON responses matching design spec shape

- [ ] **Step 1: Write tests for extended API routes**
- [ ] **Step 2: Implement all new route handlers**
- [ ] **Step 3: Run tests and verify all pass**
- [ ] **Step 4: Commit**

---

### Task 5: Full suite integration and commit

**Files:**
- Modify: `AI_Continuity.md`
- Test: Run full test suite

- [ ] **Step 1: Run full test suite (91 + new tests)**
- [ ] **Step 2: Update AI_Continuity.md**
- [ ] **Step 3: Final commit**
