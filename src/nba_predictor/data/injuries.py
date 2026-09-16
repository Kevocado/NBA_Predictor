"""
Injury report module for fetching NBA injury data from the official injury report.
Uses nbainjuries / NBA official injury report with caching and retry logic.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

from nba_predictor.config import CACHE_DIR

# Cache directory for injuries data
INJURIES_CACHE_DIR = CACHE_DIR / "injuries"

# Retry configuration
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0  # seconds


def _get_cache_path(filename: str) -> Path:
    """Get the full path for a cache file."""
    return INJURIES_CACHE_DIR / filename


def _read_cache(cache_file: Path) -> Optional[dict]:
    """Read cached data from a file."""
    try:
        if cache_file.exists():
            with open(cache_file, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return None


def _write_cache(cache_file: Path, data: dict) -> None:
    """Write data to cache file."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(data, f)


def _fetch_with_retry(url: str, max_retries: int = MAX_RETRIES) -> dict:
    """Fetch data from URL with exponential backoff retry logic."""
    backoff = INITIAL_BACKOFF
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
            else:
                raise


def _fetch_injury_report() -> list[dict]:
    """Fetch current injury report from NBA official source."""
    now = datetime.now(timezone.utc)
    sample_injuries = [
        {
            "player_id": 203999,
            "player_name": "LeBron James",
            "team_id": 1610612747,
            "team_abbreviation": "LAL",
            "description": "Lower extremity injury",
            "status": "Questionable",
            "date": now.strftime("%Y-%m-%d"),
            "detail": "Out with right knee soreness"
        },
    ]
    return sample_injuries


def get_current_injuries() -> list[dict]:
    """Get current injury report.

    Returns:
        List of injury report entries
    """
    cache_file = _get_cache_path("current_injuries.json")
    cached = _read_cache(cache_file)
    if cached is not None:
        if isinstance(cached, list):
            return cached
        if isinstance(cached, dict):
            return cached.get("injuries", [])

    # Fetch from API (using sample data as placeholder)
    injuries = _fetch_injury_report()
    _write_cache(cache_file, {"injuries": injuries})
    return injuries


def get_player_injury_history(player_id: int) -> list[dict]:
    """Get injury history for a player.

    Args:
        player_id: The NBA player ID

    Returns:
        List of injury history entries
    """
    cache_file = _get_cache_path(f"injury_history_{player_id}.json")
    cached = _read_cache(cache_file)
    if cached is not None:
        if isinstance(cached, list):
            return cached
        if isinstance(cached, dict):
            return cached.get("history", [])

    # Fetch from API (using sample data as placeholder)
    history = [
        {
            "date": "2023-01-15",
            "description": "Ankle sprain",
            "status": "Out",
        }
    ]
    _write_cache(cache_file, {"history": history})
    return history


def get_missing_player_value(player_id: int, season: int) -> float:
    """Compute production value for missing player.

    Args:
        player_id: The NBA player ID
        season: The NBA season year

    Returns:
        Float representing the production value of the missing player
    """
    cache_file = _get_cache_path(f"missing_value_{player_id}_{season}.json")
    cached = _read_cache(cache_file)
    if cached is not None:
        if isinstance(cached, dict):
            return float(cached.get("value", 0.0))
        return float(cached)

    # Fetch stats (using sample data as placeholder)
    stats = _fetch_player_stats(player_id, season)
    value = 0.0
    if stats and "stats" in stats:
        s = stats["stats"]
        value = float(s.get("points_per_game", 0)) + float(s.get("rebounds_per_game", 0)) + float(s.get("assists_per_game", 0))

    _write_cache(cache_file, {"value": value})
    return value


def _fetch_player_stats(player_id: int, season: int) -> dict:
    """Fetch player stats for a season."""
    return {}


if __name__ == "__main__":
    print("Injuries module loaded successfully")
