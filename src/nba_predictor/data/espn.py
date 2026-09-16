"""
ESPN data module for fetching NBA injury and lineup status.
Uses unofficial ESPN API endpoints.
"""

import json
import time
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from nba_predictor import config

# Base URL for the unofficial ESPN API
ESPN_API_BASE = "https://site.api.espn.com/apis/v2/sports/basketball"

# Cache directory for ESPN data
ESPN_CACHE_DIR = config.CACHE_DIR / "espn"


def _cache_path(endpoint: str, id_value: str) -> Path:
    """Generate cache file path for a given endpoint and ID."""
    safe_id = id_value.replace("/", "_").replace(":", "_")
    return ESPN_CACHE_DIR / f"{endpoint}_{safe_id}.json"


def _load_cache(endpoint: str, id_value: str) -> dict | None:
    """Load cached data if it exists and is not stale."""
    cache_file = _cache_path(endpoint, id_value)
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)
                return cached
        except (json.JSONDecodeError, IOError):
            return None
    return None


def _save_cache(endpoint: str, id_value: str, data: dict) -> None:
    """Save data to cache."""
    cache_file = _cache_path(endpoint, id_value)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=32),
    reraise=True,
)
def _fetch_espn_data(url: str) -> dict:
    """Fetch data from ESPN API with exponential backoff retry."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def get_injuries(game_id: str) -> dict:
    """Get injury status for a game from ESPN.

    Args:
        game_id: The ESPN game ID

    Returns:
        Dict containing injury information with keys: gameId, injuries
    """
    # Check cache
    cached = _load_cache("injuries", game_id)
    if cached is not None:
        return cached

    url = f"{ESPN_API_BASE}/games/{game_id}/injuries"
    data = _fetch_espn_data(url)

    result = {
        "gameId": data.get("gameId", game_id),
        "injuries": data.get("injuries", []),
    }
    _save_cache("injuries", game_id, result)
    return result


def get_lineup(game_id: str) -> dict:
    """Get lineup for a game from ESPN.

    Args:
        game_id: The ESPN game ID

    Returns:
        Dict containing lineup information with keys: gameId, lineups
    """
    # Check cache
    cached = _load_cache("lineup", game_id)
    if cached is not None:
        return cached

    url = f"{ESPN_API_BASE}/games/{game_id}/lineup"
    data = _fetch_espn_data(url)

    result = {
        "gameId": data.get("gameId", game_id),
        "lineups": data.get("lineups", []),
    }
    _save_cache("lineup", game_id, result)
    return result


def get_team_status(team_id: int) -> dict:
    """Get team status (rest days, back-to-back) from ESPN.

    Args:
        team_id: The ESPN team ID

    Returns:
        Dict containing team status information
    """
    # Check cache
    cached = _load_cache("team_status", str(team_id))
    if cached is not None:
        return cached

    url = f"{ESPN_API_BASE}/teams/{team_id}/status"
    data = _fetch_espn_data(url)

    result = {
        "teamId": data.get("teamId", team_id),
        "restDays": data.get("restDays", 0),
        "backToBack": data.get("backToBack", False),
    }
    _save_cache("team_status", str(team_id), result)
    return result


if __name__ == "__main__":
    print("ESPN module loaded successfully")
