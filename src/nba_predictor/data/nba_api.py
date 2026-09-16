"""
NBA API data fetcher module.

Provides functions to fetch NBA game data, boxscores, player stats, and more
from the nba_api library with caching and retry logic.
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from nba_predictor import config


# Cache directory for nba_api data
CACHE_DIR = config.CACHE_DIR / "nba_api"

# Retry configuration
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1.0  # seconds


def _get_cache_path(endpoint: str, identifier: str, date: str) -> Path:
    """Generate cache file path for an API endpoint."""
    clean_id = str(identifier).replace("/", "_").replace("\\", "_")
    return CACHE_DIR / f"{endpoint}_{clean_id}_{date}.json"


def _load_cache(endpoint: str, identifier: str, date: str) -> Any | None:
    """Load data from cache if it exists and is valid."""
    cache_path = _get_cache_path(endpoint, identifier, date)
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _save_cache(endpoint: str, identifier: str, date: str, data: Any) -> None:
    """Save data to cache."""
    cache_path = _get_cache_path(endpoint, identifier, date)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f)


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=INITIAL_RETRY_DELAY),
    retry=retry_if_exception_type((requests.exceptions.RequestException,)),
)
def _make_request_with_retry(url: str) -> requests.Response:
    """Make an HTTP request with retry logic."""
    response = requests.get(url, timeout=30)
    if response.status_code == 429:
        time.sleep(60)
        raise requests.exceptions.HTTPError("Rate limit exceeded", response=response)
    response.raise_for_status()
    return response


def get_schedule(date: str) -> list[dict]:
    """Get NBA games for a specific date using nba_api."""
    cached = _load_cache("schedule", date, date)
    if cached is not None:
        return cached

    try:
        from nba_api.stats.endpoints import CommonBoard
        games_finder = CommonBoard(
            league_id="00",
            season="2024-25",
            season_type="Regular Season",
            sort_column="GAME_DATE",
            date_from=date,
            date_to=date,
        )
        result = games_finder.get_dict()
        games = result.get("resultSets", [])[0].get("rowSet", [])
        headers = result.get("resultSets", [])[0].get("headers", [])
        games_list = [dict(zip(headers, row)) for row in games]
        _save_cache("schedule", date, date, games_list)
        return games_list
    except Exception as e:
        print(f"Error fetching games for {date}: {e}")
        raise


def get_boxscore(game_id: str) -> dict:
    """Get boxscore for a specific game using nba_api."""
    cached = _load_cache("boxscore", game_id, "2024-10-25")
    if cached is not None:
        return cached

    try:
        from nba_api.stats.endpoints import BoxScoreTraditionalV2
        boxscore = BoxScoreTraditionalV2(game_id=game_id)
        result = boxscore.get_dict()
        _save_cache("boxscore", game_id, "2024-10-25", result)
        return result
    except Exception as e:
        print(f"Error fetching boxscore for {game_id}: {e}")
        raise


def get_four_factors(game_id: str) -> dict:
    """Get Four Factors stats for a specific game using nba_api."""
    cached = _load_cache("four_factors", game_id, "2024-10-25")
    if cached is not None:
        return cached

    try:
        from nba_api.stats.endpoints import FourFactors
        four_factors = FourFactors(game_id=game_id)
        result = four_factors.get_dict()
        _save_cache("four_factors", game_id, "2024-10-25", result)
        return result
    except Exception as e:
        print(f"Error fetching Four Factors for {game_id}: {e}")
        raise


def get_player_stats(player_id: str, season: int) -> dict:
    """Get season stats for a specific player using nba_api."""
    season_str = f"{season}-{(season % 100) + 1:02d}"
    cached = _load_cache("player_career_stats", player_id, season_str)
    if cached is not None:
        return cached

    try:
        from nba_api.stats.endpoints import PlayerCareerStats
        career = PlayerCareerStats(player_id=player_id)
        result = career.get_dict()
        _save_cache("player_career_stats", player_id, season_str, result)
        return result
    except Exception as e:
        print(f"Error fetching player stats for {player_id}: {e}")
        raise


def get_team_stats(team_id: int, season: int) -> dict:
    """Get season stats for a specific team using nba_api."""
    season_str = f"{season}-{(season % 100) + 1:02d}"
    cached = _load_cache("team_stats", str(team_id), season_str)
    if cached is not None:
        return cached

    try:
        from nba_api.stats.endpoints import TeamDashboardByYearOld
        dashboard = TeamDashboardByYearOld(
            team_id=team_id,
            season=season_str,
            season_type="Regular Season",
        )
        result = dashboard.get_dict()
        _save_cache("team_stats", str(team_id), season_str, result)
        return result
    except Exception as e:
        print(f"Error fetching team stats for {team_id}: {e}")
        raise


def get_play_by_play(game_id: str) -> list[dict]:
    """Get play-by-play data for a specific game using nba_api."""
    cached = _load_cache("play_by_play", game_id, "2024-10-25")
    if cached is not None:
        return cached

    try:
        from nba_api.stats.endpoints import PlayByPlay
        pbp = PlayByPlay(game_id=game_id)
        result = pbp.get_dict()
        actions = result.get("resultSets", [])[0].get("rowSet", [])
        headers = result.get("resultSets", [])[0].get("headers", [])
        plays_list = [dict(zip(headers, row)) for row in actions]
        _save_cache("play_by_play", game_id, "2024-10-25", plays_list)
        return plays_list
    except Exception as e:
        print(f"Error fetching play-by-play for {game_id}: {e}")
        raise


# Module-level function aliases
get_nba_games = get_schedule
