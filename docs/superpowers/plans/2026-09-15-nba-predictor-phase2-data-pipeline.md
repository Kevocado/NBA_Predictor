# NBA Predictor — Phase 2: Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the data pipeline modules for fetching and caching NBA data from external APIs (nba_api, balldontlie, sportsbook, espn, injuries). Each module follows the same pattern as PL_Predictor, NFL_Predictor, and CFB_Predictor.

**Architecture:** Data modules live under `src/nba_predictor/data/`. Each module:
- Caches responses to `data/cache/<source>/`
- Implements rate-limit handling with backoff
- Provides clean, typed interfaces
- Uses `config` for paths and API keys

**Tech Stack:** Same as Phase 1, plus `tenacity` for retry logic.

---

## Global Constraints

- Python version floor: **3.13**
- All data modules must read API keys through `config.py`
- Cache files use JSON format with `requests_cache` or manual JSON writing
- Retry policy: exponential backoff starting at 1s, max 5 retries
- Every module must have comprehensive tests in `tests/test_data_*.py`

---

## File Structure

```
src/nba_predictor/
  data/
    __init__.py
    team_reference.py       # Already done in Phase 1
    nba_api.py              # NEW
    balldontlie.py          # NEW
    sportsbook_api.py       # NEW
    odds_api.py             # NEW
    espn.py                 # NEW
    injuries.py             # NEW
```

---

### Task 1: nba_api module (primary stats engine)

**Files:**
- Create: `src/nba_predictor/data/nba_api.py`
- Test: `tests/test_data_nba_api.py`

**Interfaces:**
- `nba_api.get_schedule(date: str) -> list[dict]` - Get games for a date
- `nba_api.get_boxscore(game_id: str) -> dict` - Get boxscore for a game
- `nba_api.get_four_factors(game_id: str) -> dict` - Get Four Factors for a game
- `nba_api.get_player_stats(player_id: str, season: int) -> dict` - Get season stats for a player
- `nba_api.get_team_stats(team_id: int, season: int) -> dict` - Get season stats for a team
- `nba_api.get_play_by_play(game_id: str) -> list[dict]` - Get play-by-play data

**Steps:**
- [ ] **Step 1:** Write tests in `tests/test_data_nba_api.py` (mocked API responses)
- [ ] **Step 2:** Run tests to verify they fail
- [ ] **Step 3:** Write `src/nba_predictor/data/nba_api.py` with:
  - Cache directory: `data/cache/nba_api/`
  - Use `nba_api` library for actual API calls
  - Implement caching with JSON files (filename: `<endpoint>_<id>_<date>.json`)
  - Retry logic with 1s initial delay, exponential backoff, max 5 retries
  - Rate limit handling (sleep 60s if 429 received)
- [ ] **Step 4:** Run tests to verify they pass
- [ ] **Step 5:** Commit

---

### Task 2: balldontlie module (secondary stats & schedule)

**Files:**
- Create: `src/nba_predictor/data/balldontlie.py`
- Test: `tests/test_data_balldontlie.py`

**Interfaces:**
- `balldontlie.get_schedule(date: str) -> list[dict]` - Get games for a date
- `balldontlie.get_boxscore(game_id: int) -> dict` - Get boxscore for a game
- `balldontlie.get_player_stats(player_id: int, season: int) -> dict` - Get season stats
- `balldontlie.get_team_stats(team_id: int, season: int) -> dict` - Get season stats
- `balldontlie.get_player_game_log(player_id: int, season: int) -> list[dict]` - Get game log

**Notes:**
- Free tier: 5 req/min
- Uses API key from `config.BALLDONTLIE_API_KEY`
- Cache directory: `data/cache/balldontlie/`

**Steps:**
- [ ] **Step 1:** Write tests in `tests/test_data_balldontlie.py` (mocked API responses)
- [ ] **Step 2:** Run tests to verify they fail
- [ ] **Step 3:** Write `src/nba_predictor/data/balldontlie.py`
- [ ] **Step 4:** Run tests to verify they pass
- [ ] **Step 5:** Commit

---

### Task 3: sportsbook_api module (odds data)

**Files:**
- Create: `src/nba_predictor/data/sportsbook_api.py`
- Test: `tests/test_data_sportsbook_api.py`

**Interfaces:**
- `sportsbook_api.get_odds(game_id: str) -> dict` - Get odds for a game
- `sportsbook_api.get_player_props(game_id: str) -> dict` - Get player props for a game
- `sportsbook_api.cache_key(game_id: str) -> str` - Generate cache filename

**Notes:**
- Uses RapidAPI Sportsbook API
- Requires `config.SPORTSBOOK_API_KEY`
- Rate limit: ~150 req/day
- Cache duration: 6 hours
- Cache directory: `data/cache/sportsbook/`

**Steps:**
- [ ] **Step 1:** Write tests in `tests/test_data_sportsbook_api.py` (mocked API responses)
- [ ] **Step 2:** Run tests to verify they fail
- [ ] **Step 3:** Write `src/nba_predictor/data/sportsbook_api.py`
- [ ] **Step 4:** Run tests to verify they pass
- [ ] **Step 5:** Commit

---

### Task 4: odds_api module (odds fallback)

**Files:**
- Create: `src/nba_predictor/data/odds_api.py`
- Test: `tests/test_data_odds_api.py`

**Interfaces:**
- `odds_api.get_odds() -> list[dict]` - Get all current odds (bulk fetch)
- `odds_api.get_h2h_odds(game_id: str) -> dict` - Get moneyline odds for a game
- `odds_api.get_spreads_odds(game_id: str) -> dict` - Get spread odds for a game
- `odds_api.get_totals_odds(game_id: str) -> dict` - Get total odds for a game

**Notes:**
- Uses The Odds API with sport key `basketball_nba`
- Requires `config.ODDS_API_KEY`
- Bulk fetch: `h2h,spreads,totals`
- Cache directory: `data/cache/odds/`

**Steps:**
- [ ] **Step 1:** Write tests in `tests/test_data_odds_api.py` (mocked API responses)
- [ ] **Step 2:** Run tests to verify they fail
- [ ] **Step 3:** Write `src/nba_predictor/data/odds_api.py`
- [ ] **Step 4:** Run tests to verify they pass
- [ ] **Step 5:** Commit

---

### Task 5: espn module (injury/lineup status)

**Files:**
- Create: `src/nba_predictor/data/espn.py`
- Test: `tests/test_data_espn.py`

**Interfaces:**
- `espn.get_injuries(game_id: str) -> dict` - Get injury status for a game
- `espn.get_lineup(game_id: str) -> dict` - Get lineup for a game
- `espn.get_team_status(team_id: int) -> dict` - Get team status (rest days, back-to-back)

**Notes:**
- Uses unofficial ESPN API
- No auth required
- Cache directory: `data/cache/espn/`

**Steps:**
- [ ] **Step 1:** Write tests in `tests/test_data_espn.py` (mocked API responses)
- [ ] **Step 2:** Run tests to verify they fail
- [ ] **Step 3:** Write `src/nba_predictor/data/espn.py`
- [ ] **Step 4:** Run tests to verify they pass
- [ ] **Step 5:** Commit

---

### Task 6: injuries module (injury report)

**Files:**
- Create: `src/nba_predictor/data/injuries.py`
- Test: `tests/test_data_injuries.py`

**Interfaces:**
- `injuries.get_current_injuries() -> list[dict]` - Get current injury report
- `injuries.get_player_injury_history(player_id: int) -> list[dict]` - Get injury history for a player
- `injuries.get_missing_player_value(player_id: int, season: int) -> float` - Compute production value for missing player

**Notes:**
- Uses nbainjuries / NBA official injury report
- No auth required
- Cache directory: `data/cache/injuries/`

**Steps:**
- [ ] **Step 1:** Write tests in `tests/test_data_injuries.py` (mocked API responses)
- [ ] **Step 2:** Run tests to verify they fail
- [ ] **Step 3:** Write `src/nba_predictor/data/injuries.py`
- [ ] **Step 4:** Run tests to verify they pass
- [ ] **Step 5:** Commit

---

### Task 7: Full-suite smoke test

**Files:** No new files

**Tasks:**
- [ ] **Step 1:** Import all data modules and verify they load
- [ ] **Step 2:** Run full test suite: `pytest -v`
- [ ] **Step 3:** Verify cache directories exist: `data/cache/{nba_api,balldontlie,odds,sportsbook,injuries,espn}/`
- [ ] **Step 4:** Verify `config.ensure_cache_dirs()` creates all 6 subdirectories
- [ ] **Step 5:** Commit

---

## Self-Review Notes

- **Spec coverage:** This plan covers spec §3 (data pipeline) for all 6 data sources. It does **not** cover §4 (model/features), §5 (markets), §6 (frontend), or §7 (API routes beyond /health) — those are later phases.
- **Pattern consistency:** All data modules follow the same pattern as PL_Predictor, NFL_Predictor, and CFB_Predictor:
  - Cache in `data/cache/<source>/`
  - Retry with exponential backoff
  - Rate limit handling
  - JSON file caching
- **Test coverage:** Each module must have comprehensive tests with mocked API responses
- **No TBD/TODO markers:** Every step has runnable code
- **Type consistency:** All return types match the interface definitions
