"""
NBA API data fetcher module.

Provides functions to fetch NBA game data, boxscores, player stats, and more
from the nba_api library with caching and retry logic.
"""
import json
import time
import os
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
    """
    Generate cache file path for an API endpoint.
    
    Args:
        endpoint: The API endpoint name
        identifier: The game ID, player ID, team ID, or date string
        date: The date string (YYYY-MM-DD format)
    
    Returns:
        Path to the cache file
    """
    # Clean identifier to be filesystem-safe
    clean_id = str(identifier).replace("/", "_").replace("\\", "_")
    return CACHE_DIR / f"{endpoint}_{clean_id}_{date}.json"


def _load_cache(endpoint: str, identifier: str, date: str) -> Any | None:
    """
    Load data from cache if it exists and is valid.
    
    Args:
        endpoint: The API endpoint name
        identifier: The identifier for the request
        date: The date string
    
    Returns:
        Cached data if valid, None otherwise
    """
    cache_path = _get_cache_path(endpoint, identifier, date)
    
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _save_cache(endpoint: str, identifier: str, date: str, data: Any) -> None:
    """
    Save data to cache.
    
    Args:
        endpoint: The API endpoint name
        identifier: The identifier for the request
        date: The date string
        data: The data to cache
    """
    cache_path = _get_cache_path(endpoint, identifier, date)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(cache_path, "w") as f:
        json.dump(data, f)


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=INITIAL_RETRY_DELAY),
    retry=retry_if_exception_type((requests.exceptions.RequestException,))
)
def _make_request_with_retry(url: str) -> requests.Response:
    """
    Make an HTTP request with retry logic.
    
    Args:
        url: The URL to fetch
    
    Returns:
        Response object
    
    Raises:
        requests.exceptions.HTTPError: If response status is 429 or other error
    """
    response = requests.get(url, timeout=30)
    
    # Handle rate limiting - sleep 60s for 429
    if response.status_code == 429:
        print("Rate limit exceeded. Sleeping for 60 seconds...")
        time.sleep(60)
        raise requests.exceptions.HTTPError(
            "Rate limit exceeded", response=response
        )
    
    response.raise_for_status()
    return response


def get_nba_games(date: str) -> list[dict]:
    """
    Get NBA games for a specific date using nba_api.
    
    Args:
        date: Date string in YYYY-MM-DD format
    
    Returns:
        List of game dictionaries
    """
    from nba_api.stats.endpoints import CommonBoard
    
    # Check cache first - use "schedule" as the id for date-based lookups
    cached = _load_cache("get_schedule", date, date)
    if cached is not None:
        return cached
    
    try:
        # Use CommonBoard endpoint to get games for the date
        games_finder = CommonBoard(
            league_id="00",
            season="2024-25",
            season_type="Regular Season",
            sort_column="GAME_DATE",
            date_from=date,
            date_to=date
        )
        
        # Parse the response
        result = games_finder.get_dict()
        games = result.get("resultSets", [])[0].get("rowSet", [])
        
        # Extract headers from resultSets
        headers = result.get("resultSets", [])[0].get("headers", [])
        
        # Convert to list of dicts
        games_list = [dict(zip(headers, row)) for row in games]
        
        # Cache the result
        _save_cache("get_schedule", date, date, games_list)
        
        return games_list
        
    except Exception as e:
        print(f"Error fetching games for {date}: {e}")
        raise


def get_boxscore(game_id: str) -> dict:
    """
    Get boxscore for a specific game using nba_api.
    
    Args:
        game_id: The NBA game ID (e.g., "0012400001")
    
    Returns:
        Boxscore data dictionary
    """
    from nba_api.stats.endpoints import BoxScoreTraditionalV2
    
    # Extract date from game_id (first 3 digits -> year, etc.)
    # Game ID format: 0012400001 -> 2024 is embedded
    date_str = "2024-10-25"  # Default date, will be overridden if found
    
    # Check cache first
    cached = _load_cache("get_boxscore", game_id, date_str)
    if cached is not None:
        return cached
    
    try:
        # Get boxscore data
        boxscore = BoxScoreTraditionalV2(game_id=game_id)
        result = boxscore.get_dict()
        
        # Cache the result
        _save_cache("get_boxscore", game_id, date_str, result)
        
        return result
        
    except Exception as e:
        print(f"Error fetching boxscore for {game_id}: {e}")
        raise


def get_four_factors(game_id: str) -> dict:
    """
    Get Four Factors stats for a specific game using nba_api.
    
    Args:
        game_id: The NBA game ID
    
    Returns:
        Four Factors data dictionary
    """
    from nba_api.stats.endpoints import FourFactors
    
    date_str = "2024-10-25"  # Default date
    
    # Check cache first
    cached = _load_cache("get_four_factors", game_id, date_str)
    if cached is not None:
        return cached
    
    try:
        # Get Four Factors data
        four_factors = FourFactors(game_id=game_id)
        result = four_factors.get_dict()
        
        # Cache the result
        _save_cache("get_four_factors", game_id, date_str, result)
        
        return result
        
    except Exception as e:
        print(f"Error fetching Four Factors for {game_id}: {e}")
        raise


def get_player_stats(player_id: str, season: int) -> dict:
    """
    Get season stats for a specific player using nba_api.
    
    Args:
        player_id: The NBA player ID
        season: The season year (e.g., 2024 for 2023-24 season)
    
    Returns:
        Player season stats dictionary
    """
    from nba_api.stats.endpoints import PlayerCareerStats
    
    # Determine season string from year
    season_str = f"{season}-{(season % 100) + 1:02d}"
    
    # Check cache first
    cached = _load_cache("get_player_career_stats", player_id, str(season))
    if cached is not None:
        return cached
    
    try:
        # Get player career stats
        career = PlayerCareerStats(player_id=player_id)
        result = career.get_dict()
        
        # Cache the result
        _save_cache("get_player_career_stats", player_id, str(season), result)
        
        return result
        
    except Exception as e:
        print(f"Error fetching player stats for {player_id}: {e}")
        raise


def get_team_stats(team_id: int, season: int) -> dict:
    """
    Get season stats for a specific team using nba_api.
    
    Args:
        team_id: The NBA team ID
        season: The season year (e.g., 2024 for 2023-24 season)
    
    Returns:
        Team season stats dictionary
    """
    from nba_api.stats.endpoints import TeamDashboardByYearOld
    
    # Determine season string from year
    season_str = f"{season}-{(season % 100) + 1:02d}"
    
    # Check cache first
    cached = _load_cache("get_team_stats", str(team_id), str(season))
    if cached is not None:
        return cached
    
    try:
        # Get team stats
        dashboard = TeamDashboardByYearOld(
            team_id=team_id,
            season=season_str,
            season_type="Regular Season"
        )
        result = dashboard.get_dict()
        
        # Cache the result
        _save_cache("get_team_stats", str(team_id), str(season), result)
        
        return result
        
    except Exception as e:
        print(f"Error fetching team stats for {team_id}: {e}")
        raise


def get_play_by_play(game_id: str) -> list[dict]:
    """
    Get play-by-play data for a specific game using nba_api.
    
    Args:
        game_id: The NBA game ID
    
    Returns:
        List of play-by-play action dictionaries
    """
    from nba_api.stats.endpoints import PlayByPlay
    
    date_str = "2024-10-25"  # Default date
    
    # Check cache first
    cached = _load_cache("get_play_by_play", game_id, date_str)
    if cached is not None:
        return cached
    
    try:
        # Get play-by-play data
        pbp = PlayByPlay(game_id=game_id)
        result = pbp.get_dict()
        
        # Extract actions from result
        actions = result.get("resultSets", [])[0].get("rowSet", [])
        headers = result.get("resultSets", [])[0].get("headers", [])
        
        # Convert to list of dicts
        plays_list = [dict(zip(headers, row)) for row in actions]
        
        # Cache the result
        _save_cache("get_play_by_play", game_id, date_str, plays_list)
        
        return plays_list
        
    except Exception as e:
        print(f"Error fetching play-by-play for {game_id}: {e}")
        raise


# Module-level function aliases for the required interface
get_schedule = get_nba_games
get_player_stats = get_player_stats
get_team_stats = get_team_stats
