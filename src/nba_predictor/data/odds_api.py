"""The Odds API integration module for fetching NBA odds data.

Provides functions to fetch odds data from The Odds API with caching and retry logic.
Uses sport key 'basketball_nba' and supports bulk fetch of h2h, spreads, and totals markets.
"""
import json
import os
import time
from pathlib import Path
from typing import Optional

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

from nba_predictor import config

# Constants
SPORT_KEY = "basketball_nba"
MARKETS = "h2h,spreads,totals"
CACHE_DIR = config.CACHE_DIR / "odds"
RETRY_STATUSES = {429, 500, 502, 503, 504}


def _retry_if_http_error(exception: Exception) -> bool:
    """Retry on specific HTTP status codes."""
    if isinstance(exception, requests.exceptions.HTTPError):
        return exception.response.status_code in RETRY_STATUSES
    return False


# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_api_key() -> str:
    """Get the Odds API key from config."""
    api_key = config.ODDS_API_KEY
    if not api_key:
        raise ValueError("ODDS_API_KEY environment variable is not set")
    return api_key


def _cache_key(game_id: Optional[str] = None, use_bulk: bool = False) -> str:
    """Generate cache filename."""
    if use_bulk:
        return "bulk_odds.json"
    return f"game_{game_id}.json" if game_id else "odds.json"


def _save_to_cache(data: dict, filename: str) -> Path:
    """Save data to cache file."""
    cache_path = CACHE_DIR / filename
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)
    return cache_path


def _load_from_cache(filename: str) -> Optional[dict]:
    """Load data from cache file if it exists and is not expired."""
    cache_path = CACHE_DIR / filename
    if not cache_path.exists():
        return None

    # Check if cache is expired (1 hour)
    cache_age = time.time() - cache_path.stat().st_mtime
    if cache_age > 3600:
        return None

    with open(cache_path, "r") as f:
        return json.load(f)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type((requests.exceptions.HTTPError, requests.exceptions.ConnectionError)),
    reraise=True,
)
def _bulk_fetch_odds() -> list[dict]:
    """Bulk fetch all odds data from The Odds API."""
    api_key = _get_api_key()
    response = requests.get(
        "https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": MARKETS,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        },
        timeout=30,
    )
    # Handle 429 specifically
    if response.status_code == 429:
        time.sleep(60)
        raise requests.exceptions.HTTPError("Rate limit exceeded", response=response)
    response.raise_for_status()
    return response.json()


def get_odds() -> list[dict]:
    """Get all current odds (bulk fetch).

    Fetches head-to-head, spread, and total odds for all active NBA games.
    Results are cached for 1 hour.

    Returns:
        list[dict]: List of odds data for each game.

    Raises:
        ValueError: If ODDS_API_KEY is not set
        requests.exceptions.RequestException: If API request fails after retries
    """
    # Try to load from cache
    cache_file = _cache_key(use_bulk=True)
    cached = _load_from_cache(cache_file)
    if cached is not None:
        return cached.get("data", cached)

    # Fetch from API
    odds_data = _bulk_fetch_odds()

    # Save to cache
    _save_to_cache({"data": odds_data, "cached_at": time.time()}, cache_file)

    return odds_data


def _get_game_odds(game_id: str, market: str) -> dict:
    """Get odds for a specific game and market type."""
    # Try to load from cache
    cache_file = _cache_key(game_id=game_id)
    cached = _load_from_cache(cache_file)

    if cached and market in cached:
        return cached

    # Fetch all odds and filter
    all_odds = get_odds()

    # Find the game
    game_odds = None
    for game in all_odds:
        if game.get("id") == game_id:
            game_odds = game
            break

    if not game_odds:
        return {}

    # Extract specific market
    result = {}
    for bookmaker in game_odds.get("bookmakers", []):
        for market_data in bookmaker.get("markets", []):
            if market_data.get("key") == market:
                result[market] = market_data
                break

    # Save to cache
    if result:
        existing = cached if cached else {}
        existing.update(result)
        _save_to_cache(existing, cache_file)

    return result


def get_h2h_odds(game_id: str) -> dict:
    """Get moneyline (head-to-head) odds for a game."""
    return _get_game_odds(game_id, "h2h")


def get_spreads_odds(game_id: str) -> dict:
    """Get spread odds for a game."""
    return _get_game_odds(game_id, "spreads")


def get_totals_odds(game_id: str) -> dict:
    """Get total (over/under) odds for a game."""
    return _get_game_odds(game_id, "totals")


def clear_cache() -> None:
    """Clear all cached odds data."""
    for cache_file in CACHE_DIR.glob("*.json"):
        cache_file.unlink()


def get_cache_info() -> dict:
    """Get information about cached odds data."""
    cache_files = list(CACHE_DIR.glob("*.json"))
    total_size = sum(f.stat().st_size for f in cache_files)
    return {
        "file_count": len(cache_files),
        "total_size_bytes": total_size,
        "cache_dir": str(CACHE_DIR),
    }
