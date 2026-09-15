"""
Sportsbook API module for fetching NBA odds data from RapidAPI Sportsbook API.

Provides:
- get_odds(game_id: str) -> dict: Get odds for a game
- get_player_props(game_id: str) -> dict: Get player props for a game
- cache_key(game_id: str) -> str: Generate cache filename

Rate limit: ~150 req/day
Cache duration: 6 hours
Cache directory: data/cache/sportsbook/
Retry policy: exponential backoff starting at 1s, max 5 retries
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from nba_predictor import config


# Cache duration in seconds (6 hours)
CACHE_DURATION_SECONDS = 6 * 60 * 60

# Cache directory for sportsbook data
CACHE_DIR = config.CACHE_DIR / "sportsbook"


def cache_key(game_id: str, endpoint: str = "odds") -> str:
    """
    Generate cache filename for a game.

    Args:
        game_id: The NBA game ID (e.g., "0021900001")
        endpoint: The API endpoint type ("odds" or "player_props")

    Returns:
        Cache filename string
    """
    return f"{endpoint}_{game_id}.json"


def _get_cache_path(game_id: str, endpoint: str = "odds") -> Path:
    """Get the full path to a cache file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / cache_key(game_id, endpoint)


def _is_cache_fresh(cache_path: Path) -> bool:
    """
    Check if a cache file exists and is still fresh (not expired).

    Args:
        cache_path: Path to the cache file

    Returns:
        True if cache exists and is still valid
    """
    if not cache_path.exists():
        return False

    try:
        with open(cache_path, "r") as f:
            data = json.load(f)

        cached_at = data.get("cached_at")
        if cached_at is None:
            return False

        # Check if cache is less than 6 hours old
        cache_time = datetime.fromtimestamp(cached_at)
        now = datetime.now()
        return (now - cache_time).total_seconds() < CACHE_DURATION_SECONDS
    except (json.JSONDecodeError, OSError, TypeError):
        return False


def _load_cache(game_id: str, endpoint: str = "odds") -> dict | None:
    """
    Load data from cache if available and fresh.

    Args:
        game_id: The NBA game ID
        endpoint: The API endpoint type

    Returns:
        Cached data dict or None if cache is stale/missing
    """
    cache_path = _get_cache_path(game_id, endpoint)
    if _is_cache_fresh(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return None


def _save_cache(game_id: str, data: dict, endpoint: str = "odds") -> None:
    """
    Save data to cache.

    Args:
        game_id: The NBA game ID
        data: The data to cache
        endpoint: The API endpoint type
    """
    cache_path = _get_cache_path(game_id, endpoint)
    data["cached_at"] = datetime.now().timestamp()
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)


def _fetch_odds_api(game_id: str) -> dict:
    """
    Fetch odds data from RapidAPI Sportsbook API.

    Args:
        game_id: The NBA game ID

    Returns:
        Odds data dictionary

    Raises:
        requests.exceptions.RequestException: If API call fails
    """
    api_key = config.SPORTSBOOK_API_KEY
    if not api_key:
        raise ValueError("SPORTSBOOK_API_KEY not set in config")

    # RapidAPI Sportsbook API endpoint
    url = "https://sportsbook-api.p.rapidapi.com/odds/nba"
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "sportsbook-api.p.rapidapi.com",
    }

    # Query params for the specific game
    params = {"game_id": game_id}

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()

    return response.json()


def _fetch_player_props_api(game_id: str) -> dict:
    """
    Fetch player props data from RapidAPI Sportsbook API.

    Args:
        game_id: The NBA game ID

    Returns:
        Player props data dictionary

    Raises:
        requests.exceptions.RequestException: If API call fails
    """
    api_key = config.SPORTSBOOK_API_KEY
    if not api_key:
        raise ValueError("SPORTSBOOK_API_KEY not set in config")

    # RapidAPI Sportsbook API endpoint for player props
    url = "https://sportsbook-api.p.rapidapi.com/props/nba"
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "sportsbook-api.p.rapidapi.com",
    }

    # Query params for the specific game
    params = {"game_id": game_id}

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()

    return response.json()


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, OSError)),
)
def _fetch_with_retry(game_id: str, endpoint: str = "odds") -> dict:
    """
    Fetch data from API with retry logic.

    Args:
        game_id: The NBA game ID
        endpoint: The API endpoint type ("odds" or "player_props")

    Returns:
        API response data

    Raises:
        requests.exceptions.RequestException: If all retries fail
    """
    if endpoint == "odds":
        return _fetch_odds_api(game_id)
    else:
        return _fetch_player_props_api(game_id)


def get_odds(game_id: str) -> dict:
    """
    Get odds for a game.

    Fetches moneyline, spread, and total odds for the specified game.
    Uses caching to avoid exceeding rate limits.

    Args:
        game_id: The NBA game ID (e.g., "0021900001")

    Returns:
        Dictionary containing:
        - game_id: The game identifier
        - home_team: Home team abbreviation
        - away_team: Away team abbreviation
        - odds: Dictionary with moneyline, spread, and total
        - last_updated: ISO format timestamp of last update

    Raises:
        ValueError: If API key is not configured
        requests.exceptions.RequestException: If all retries fail
    """
    # Check cache first
    cached = _load_cache(game_id, "odds")
    if cached is not None:
        return cached

    # Fetch from API with retry logic
    data = _fetch_with_retry(game_id, "odds")

    # Save to cache
    _save_cache(game_id, data, "odds")

    return data


def get_player_props(game_id: str) -> dict:
    """
    Get player props for a game.

    Fetches player proposition bets for the specified game.
    Uses caching to avoid exceeding rate limits.

    Args:
        game_id: The NBA game ID (e.g., "0021900001")

    Returns:
        Dictionary containing:
        - game_id: The game identifier
        - player_props: List of player proposition objects, each containing:
          - player_id: The NBA player ID
          - player_name: Full player name
          - team: Team abbreviation
          - prop_type: Type of prop (points, rebounds, assists, etc.)
          - odds: Dictionary with over/under lines and odds

    Raises:
        ValueError: If API key is not configured
        requests.exceptions.RequestException: If all retries fail
    """
    # Check cache first
    cached = _load_cache(game_id, "player_props")
    if cached is not None:
        return cached

    # Fetch from API with retry logic
    data = _fetch_with_retry(game_id, "player_props")

    # Save to cache
    _save_cache(game_id, data, "player_props")

    return data
