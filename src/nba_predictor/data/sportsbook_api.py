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
    """Generate cache filename for a game."""
    return f"{endpoint}_{game_id}.json"


def _get_cache_path(game_id: str, endpoint: str = "odds") -> Path:
    """Get the full path to a cache file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / cache_key(game_id, endpoint)


def _is_cache_fresh(cache_path: Path) -> bool:
    """Check if a cache file exists and is still fresh (not expired)."""
    if not cache_path.exists():
        return False
    try:
        with open(cache_path, "r") as f:
            data = json.load(f)
        cached_at = data.get("cached_at")
        if cached_at is None:
            return False
        cache_time = datetime.fromtimestamp(cached_at)
        now = datetime.now()
        return (now - cache_time).total_seconds() < CACHE_DURATION_SECONDS
    except (json.JSONDecodeError, OSError, TypeError):
        return False


def _load_cache(game_id: str, endpoint: str = "odds") -> dict | None:
    """Load data from cache if available and fresh."""
    cache_path = _get_cache_path(game_id, endpoint)
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r") as f:
            data = json.load(f)
        if _is_cache_fresh(cache_path):
            return data
    except (json.JSONDecodeError, IOError):
        pass
    return None


def _save_cache(game_id: str, endpoint: str, data: dict) -> None:
    """Save data to cache."""
    cache_path = _get_cache_path(game_id, endpoint)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data_with_timestamp = {**data, "cached_at": time.time()}
    with open(cache_path, "w") as f:
        json.dump(data_with_timestamp, f, indent=2)


def _get_api_key() -> str:
    """Get the Sportsbook API key from config."""
    api_key = config.SPORTSBOOK_API_KEY
    if not api_key:
        raise ValueError("SPORTSBOOK_API_KEY not set")
    return api_key


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type((requests.exceptions.HTTPError, requests.exceptions.ConnectionError)),
)
def _fetch_odds_api(game_id: str) -> dict:
    """Fetch odds from the Sportsbook API."""
    api_key = _get_api_key()
    url = f"https://api.the-odds-api.com/sportsbook/v1/odds/{game_id}"
    response = requests.get(url, headers={"Authorization": api_key}, timeout=30)
    response.raise_for_status()
    return response.json()


def get_odds(game_id: str) -> dict:
    """Get odds for a game."""
    # Try cache first
    cached = _load_cache(game_id, "odds")
    if cached is not None:
        return cached.get("data", cached)

    # Fetch from API
    odds_data = _fetch_odds_api(game_id)
    _save_cache(game_id, "odds", odds_data)
    return odds_data


def get_player_props(game_id: str) -> dict:
    """Get player props for a game."""
    cached = _load_cache(game_id, "player_props")
    if cached is not None:
        return cached.get("data", cached)

    api_key = _get_api_key()
    url = f"https://api.the-odds-api.com/sportsbook/v1/player_props/{game_id}"
    response = requests.get(url, headers={"Authorization": api_key}, timeout=30)
    response.raise_for_status()
    props_data = response.json()
    _save_cache(game_id, "player_props", props_data)
    return props_data


if __name__ == "__main__":
    print("Sportsbook API module loaded successfully")
