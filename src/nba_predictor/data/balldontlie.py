"""
BallDontLie API client module for fetching NBA data.

This module provides interfaces to fetch NBA data from the BallDontLie API
with caching and retry logic.

Free tier: 5 req/min
API key: config.BALLDONTLIE_API_KEY
Cache directory: data/cache/balldontlie/
"""
import json
import time
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
    global _CACHE_DIR
    _CACHE_DIR = config.CACHE_DIR / "balldontlie"
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR

# Module-level cache dir (can be patched in tests)
_CACHE_DIR = config.CACHE_DIR / "balldontlie"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _make_request(endpoint: str, params: dict | None = None) -> dict:
    """Make a request to the BallDontLie API with retry logic and rate limit handling."""
    api_key = _get_api_key()
    url = f"{BASE_URL}/{endpoint}"
    headers = {"Authorization": api_key}

    backoff = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            if response.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RATE_LIMIT_WAIT)
                    continue
                else:
                    response.raise_for_status()
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
    raise requests.exceptions.RequestException("Max retries exceeded")


def _cache_path(endpoint: str, **kwargs) -> Path:
    """Generate a cache file path based on endpoint and parameters."""
    filename_parts = [endpoint]
    for key, value in sorted(kwargs.items()):
        filename_parts.append(f"{key}_{value}")
    filename = "_".join(filename_parts) + ".json"
    return _CACHE_DIR / filename


def _load_from_cache(endpoint: str, **kwargs) -> dict | None:
    """Load cached data if available."""
    cache_file = _cache_path(endpoint, **kwargs)
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def _save_to_cache(endpoint: str, data: dict, **kwargs) -> None:
    """Save data to cache."""
    cache_file = _cache_path(endpoint, **kwargs)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


def get_schedule(date: str) -> list[dict]:
    """Get games for a specific date."""
    endpoint = "games"
    params = {"dates[]": date}

    cached = _load_from_cache(endpoint, date=date)
    if cached:
        return cached.get("data", [])

    response = _make_request(endpoint, params)
    _save_to_cache(endpoint, response, date=date)
    return response.get("data", [])


def get_boxscore(game_id: int) -> dict:
    """Get boxscore for a specific game."""
    endpoint = f"box_scores/{game_id}"

    cached = _load_from_cache(endpoint, game_id=game_id)
    if cached:
        return cached

    response = _make_request(endpoint)
    _save_to_cache(endpoint, response, game_id=game_id)
    return response


def get_player_stats(player_id: int, season: int) -> dict:
    """Get season stats for a player."""
    endpoint = "stats"
    params = {"player_ids[]": player_id, "seasons[]": season}

    cached = _load_from_cache(endpoint, player_id=player_id, season=season)
    if cached:
        data = cached.get("data", [])
        return data[0] if data else {}

    response = _make_request(endpoint, params)
    _save_to_cache(endpoint, response, player_id=player_id, season=season)
    data = response.get("data", [])
    return data[0] if data else {}


def get_team_stats(team_id: int, season: int) -> dict:
    """Get season stats for a specific team."""
    endpoint = "teams"
    params = {"seasons[]": season}

    cached = _load_from_cache(endpoint, team_id=team_id, season=season)
    if cached:
        data = cached.get("data", [])
        return data[0] if data else {}

    try:
        response = _make_request(endpoint, params)
    except Exception:
        return {}

    teams = response.get("data", [])
    for team in teams:
        if team.get("id") == team_id or team.get("team_id") == team_id:
            _save_to_cache("team_stats", {"data": team}, team_id=team_id, season=season)
            return team

    return {}


def get_player_game_log(player_id: int, season: int) -> list[dict]:
    """Get game log for a player in a season."""
    endpoint = "stats"
    params = {"player_ids[]": player_id, "seasons[]": season}

    cached = _load_from_cache(endpoint, player_id=player_id, season=season)
    if cached:
        return cached.get("data", [])

    response = _make_request(endpoint, params)
    _save_to_cache(endpoint, response, player_id=player_id, season=season)
    return response.get("data", [])
