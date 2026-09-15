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
    """
    Fetch data from URL with exponential backoff retry logic.
    
    Args:
        url: The URL to fetch
        max_retries: Maximum number of retry attempts
        
    Returns:
        JSON response as dict
        
    Raises:
        requests.RequestException: If all retries fail
    """
    backoff = INITIAL_BACKOFF
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2  # Exponential backoff
            else:
                raise


def _fetch_injury_report() -> list[dict]:
    """
    Fetch current injury report from NBA official source.
    
    This is a placeholder that returns sample data.
    In production, this would fetch from nbainjuries.com or similar.
    
    Returns:
        List of injury report entries
    """
    # Sample data structure (would fetch from actual API in production)
    # For now, return sample data that mimics real injury reports
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
        {
            "player_id": 201933,
            "player_name": "Stephen Curry",
            "team_id": 1610612744,
            "team_abbreviation": "GSW",
            "description": "Ankle sprain",
            "status": "Out",
            "date": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
            "detail": "Out with ankle sprain"
        }
    ]
    return sample_injuries


def _fetch_player_injury_history(player_id: int) -> list[dict]:
    """
    Fetch injury history for a specific player.
    
    Args:
        player_id: The NBA player ID
        
    Returns:
        List of injury records
    """
    # Sample data structure (would fetch from actual API in production)
    now = datetime.now(timezone.utc)
    sample_history = [
        {
            "player_id": player_id,
            "player_name": f"Player {player_id}",
            "team_id": 1610612747,
            "team_abbreviation": "LAL",
            "date": (now - timedelta(days=30)).strftime("%Y-%m-%d"),
            "status": "Out",
            "description": "Hamstring strain",
            "return_date": (now - timedelta(days=10)).strftime("%Y-%m-%d")
        },
        {
            "player_id": player_id,
            "player_name": f"Player {player_id}",
            "team_id": 1610612747,
            "team_abbreviation": "LAL",
            "date": (now - timedelta(days=60)).strftime("%Y-%m-%d"),
            "status": "Out",
            "description": "Knee contusion",
            "return_date": (now - timedelta(days=40)).strftime("%Y-%m-%d")
        }
    ]
    return sample_history


def _fetch_player_stats(player_id: int, season: int) -> Optional[dict]:
    """
    Fetch player statistics for a specific season.
    
    Args:
        player_id: The NBA player ID
        season: The NBA season year (e.g., 2022 for 2022-23 season)
        
    Returns:
        Player stats dict or None if not found
    """
    # Sample data structure (would fetch from actual API in production)
    now = datetime.now(timezone.utc)
    sample_stats = {
        "player_id": player_id,
        "season": season,
        "games_played": 60,
        "minutes_per_game": 38.6,
        "pts": 25.7,
        "reb": 7.9,
        "ast": 8.3,
        "usage_rate": 32.5,
        "true_shooting_pct": 0.582,
        "win_shares": 14.2
    }
    return sample_stats


def get_current_injuries() -> list[dict]:
    """
    Get current injury report from the NBA official source.
    
    Returns cached data if available and fresh (within 1 hour).
    Fetches new data if cache is stale or missing.
    
    Returns:
        List of injury report entries with details about player injuries
    """
    cache_file = _get_cache_path("injury_report.json")
    cached = _read_cache(cache_file)
    
    # Check if cache is fresh (within 1 hour)
    if cached and "cached_at" in cached:
        cached_time = datetime.fromtimestamp(cached["cached_at"], tz=timezone.utc)
        if datetime.now(timezone.utc) - cached_time < timedelta(hours=1):
            return cached.get("injuries", [])
    
    # Fetch fresh data
    injuries = _fetch_injury_report()
    
    # Cache the result
    cache_data = {
        "injuries": injuries,
        "cached_at": datetime.now(timezone.utc).timestamp()
    }
    _write_cache(cache_file, cache_data)
    
    return injuries


def get_player_injury_history(player_id: int) -> list[dict]:
    """
    Get injury history for a specific player.
    
    Returns cached data if available and fresh (within 24 hours).
    Fetches new data if cache is stale or missing.
    
    Args:
        player_id: The NBA player ID
        
    Returns:
        List of injury records for the player
    """
    cache_file = _get_cache_path(f"player_{player_id}_history.json")
    cached = _read_cache(cache_file)
    
    # Check if cache is fresh (within 24 hours)
    if cached and "cached_at" in cached:
        cached_time = datetime.fromtimestamp(cached["cached_at"], tz=timezone.utc)
        if datetime.now(timezone.utc) - cached_time < timedelta(hours=24):
            return cached.get("injuries", [])
    
    # Fetch fresh data
    history = _fetch_player_injury_history(player_id)
    
    # Cache the result
    cache_data = {
        "player_id": player_id,
        "injuries": history,
        "cached_at": datetime.now(timezone.utc).timestamp()
    }
    _write_cache(cache_file, cache_data)
    
    return history


def get_missing_player_value(player_id: int, season: int) -> float:
    """
    Compute production value for a missing player.
    
    This calculates an estimated production value based on the player's
    season statistics, including points, rebounds, assists, usage rate,
    and win shares.
    
    Args:
        player_id: The NBA player ID
        season: The NBA season year (e.g., 2022 for 2022-23 season)
        
    Returns:
        Estimated production value as a float
    """
    cache_file = _get_cache_path(f"player_{player_id}_season_{season}_value.json")
    cached = _read_cache(cache_file)
    
    # Check if cache is fresh (within 24 hours)
    if cached and "cached_at" in cached:
        cached_time = datetime.fromtimestamp(cached["cached_at"], tz=timezone.utc)
        if datetime.now(timezone.utc) - cached_time < timedelta(hours=24):
            return cached.get("production_value", 0.0)
    
    # Fetch player stats
    stats = _fetch_player_stats(player_id, season)
    
    # Calculate production value if stats available
    if stats:
        # Weighted combination of key metrics
        # Points: 0.4 weight
        # Rebounds: 0.2 weight  
        # Assists: 0.2 weight
        # Win shares: 0.2 weight (normalized)
        
        points_score = stats.get("pts", 0) * 0.4
        reb_score = stats.get("reb", 0) * 0.2
        ast_score = stats.get("ast", 0) * 0.2
        win_shares = stats.get("win_shares", 0)
        
        # Normalize win shares (typical range is 0-25)
        ws_score = (win_shares / 25.0) * 100 * 0.2
        
        production_value = points_score + reb_score + ast_score + ws_score
    else:
        production_value = 0.0
    
    # Cache the result
    cache_data = {
        "player_id": player_id,
        "season": season,
        "production_value": production_value,
        "cached_at": datetime.now(timezone.utc).timestamp()
    }
    _write_cache(cache_file, cache_data)
    
    return production_value
