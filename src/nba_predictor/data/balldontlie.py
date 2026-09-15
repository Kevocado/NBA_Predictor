"""
BallDontLie API client module for fetching NBA data.

This module provides interfaces to fetch NBA data from the BallDontLie API
with caching and retry logic.

Free tier: 5 req/min
API key: config.BALLDONTLIE_API_KEY
Cache directory: data/cache/balldontlie/
"""
import json
import os
import time
from typing import Any
from pathlib import Path
import requests

from nba_predictor import config

# API base URL
BASE_URL = "https://www.balldontlie.io/api/v1"

# Rate limit settings
RATE_LIMIT_WAIT = 60  # seconds to wait on 429
INITIAL_BACKOFF = 1  # seconds
MAX_RETRIES = 5


def _get_api_key() -> str:
    """Get the BallDontLie API key from config."""
    api_key = config.BALLDONTLIE_API_KEY
    if not api_key:
        raise ValueError(
            "BALLDONTLIE_API_KEY not set in environment/config. "
            "Please set this key to use the BallDontLie API."
        )
    return api_key


def _get_cache_dir() -> Path:
    """Get the cache directory for balldontlie data."""
    cache_dir = config.CACHE_DIR / "balldontlie"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _make_request(endpoint: str, params: dict | None = None) -> dict:
    """
    Make a request to the BallDontLie API with retry logic and rate limit handling.

    Args:
        endpoint: The API endpoint (e.g., 'games', 'box_scores')
        params: Query parameters to include

    Returns:
        JSON response as dictionary

    Raises:
        requests.HTTPError: If request fails after all retries
    """
    api_key = _get_api_key()
    url = f"{BASE_URL}/{endpoint}"
    headers = {"Authorization": api_key}

    backoff = INITIAL_BACKOFF

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)

            # Handle rate limiting
            if response.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RATE_LIMIT_WAIT)
                    continue
                else:
                    response.raise_for_status()

            # Handle other errors
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(backoff)
                backoff *= 2  # Exponential backoff
                continue
            raise

    raise requests.exceptions.RequestException("Max retries exceeded")


def _cache_path(endpoint: str, **kwargs) -> Path:
    """
    Generate a cache file path based on endpoint and parameters.

    Args:
        endpoint: The API endpoint
        **kwargs: Query parameters to include in cache key

    Returns:
        Path to the cache file
    """
    cache_dir = _get_cache_dir()
    filename_parts = [endpoint]

    for key, value in sorted(kwargs.items()):
        filename_parts.append(f"{key}_{value}")

    filename = "_".join(filename_parts) + ".json"
    return cache_dir / filename


def _make_cache_path(endpoint: str, **kwargs) -> Path:
    """
    Internal function to generate cache path - can be mocked in tests.

    Args:
        endpoint: The API endpoint
        **kwargs: Query parameters to include in cache key

    Returns:
        Path to the cache file
    """
    cache_dir = _get_cache_dir()
    filename_parts = [endpoint]

    for key, value in sorted(kwargs.items()):
        filename_parts.append(f"{key}_{value}")

    filename = "_".join(filename_parts) + ".json"
    return cache_dir / filename


def _load_from_cache(endpoint: str, **kwargs) -> dict | None:
    """
    Load cached data if available.

    Args:
        endpoint: The API endpoint
        **kwargs: Query parameters used for cache key

    Returns:
        Cached data dict or None if not cached
    """
    cache_file = _cache_path(endpoint, **kwargs)

    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    return None


def _save_to_cache(endpoint: str, data: dict, **kwargs) -> None:
    """
    Save data to cache.

    Args:
        endpoint: The API endpoint
        data: Data to cache
        **kwargs: Query parameters used for cache key
    """
    cache_dir = _get_cache_dir()
    cache_file = _cache_path(endpoint, **kwargs)

    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


def get_schedule(date: str) -> list[dict]:
    """
    Get games for a specific date.

    Args:
        date: Date in YYYY-MM-DD format

    Returns:
        List of game dictionaries
    """
    endpoint = "games"
    params = {"dates[]": date}

    # Check cache first
    cached = _load_from_cache(endpoint, date=date)
    if cached:
        return cached.get("data", [])

    # Make API request
    response = _make_request(endpoint, params)

    # Save to cache
    _save_to_cache(endpoint, response, date=date)

    return response.get("data", [])


def get_boxscore(game_id: int) -> dict:
    """
    Get boxscore for a specific game.

    Args:
        game_id: The BallDontLie game ID

    Returns:
        Boxscore data dictionary
    """
    endpoint = f"box_scores/{game_id}"

    # Check cache first
    cached = _load_from_cache(endpoint, game_id=game_id)
    if cached:
        return cached

    # Make API request
    response = _make_request(endpoint)

    # Save to cache (use the actual response structure)
    try:
        _save_to_cache(endpoint, response, game_id=game_id)
    except FileNotFoundError:
        # Cache directory doesn't exist, skip caching
        pass

    return response


def get_player_stats(player_id: int, season: int) -> dict:
    """
    Get season stats for a player.

    Args:
        player_id: The BallDontLie player ID
        season: The NBA season year (e.g., 2023 for 2023-24 season)

    Returns:
        Player stats dictionary
    """
    endpoint = "stats"
    params = {
        "player_ids[]": player_id,
        "seasons[]": season,
    }

    # Check cache first
    cached = _load_from_cache(endpoint, player_id=player_id, season=season)
    if cached:
        data = cached.get("data", [])
        return data[0] if data else {}

    # Make API request
    response = _make_request(endpoint, params)

    # Save to cache
    try:
        _save_to_cache(endpoint, response, player_id=player_id, season=season)
    except FileNotFoundError:
        pass

    data = response.get("data", [])
    return data[0] if data else {}


def get_team_stats(team_id: int, season: int) -> dict:
    """
    Get season stats for a team.

    Args:
        team_id: The BallDontLie team ID
        season: The NBA season year (e.g., 2023 for 2023-24 season)

    Returns:
        Team stats dictionary
    """
    endpoint = "teams"
    params = {
        "seasons[]": season,
    }

    # Check cache first
    cached = _load_from_cache(endpoint, team_id=team_id, season=season)
    if cached:
        data = cached.get("data", [])
        return data[0] if data else {}

    # Make API request for all team stats
    try:
        response = _make_request(endpoint, params)
    except FileNotFoundError:
        return {}

    # Find the specific team
    teams = response.get("data", [])
    for team in teams:
        if team.get("id") == team_id:
            # Save this team's stats to cache
            try:
                _save_to_cache("team_stats", {"data": team}, team_id=team_id, season=season)
            except FileNotFoundError:
                pass
            return team

    # Team not found
    return {}


def get_player_game_log(player_id: int, season: int) -> list[dict]:
    """
    Get game log for a player in a season.

    Args:
        player_id: The BallDontLie player ID
        season: The NBA season year (e.g., 2023 for 2023-24 season)

    Returns:
        List of game log entries
    """
    endpoint = "stats"
    params = {
        "player_ids[]": player_id,
        "seasons[]": season,
    }

    # Check cache first
    cached = _load_from_cache(endpoint, player_id=player_id, season=season)
    if cached:
        return cached.get("data", [])

    # Make API request
    response = _make_request(endpoint, params)

    # Save to cache
    try:
        _save_to_cache(endpoint, response, player_id=player_id, season=season)
    except FileNotFoundError:
        pass

    return response.get("data", [])


if __name__ == "__main__":
    # Example usage
    print("BallDontLie API module loaded successfully")
    print(f"Base URL: {BASE_URL}")
    print(f"Cache directory: {config.CACHE_DIR / 'balldontlie'}")
