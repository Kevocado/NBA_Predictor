"""Frozen earlier seasons, so the retrain can use more than the one-season ingest window.

Walk-forward over the 2019-2025 openers (docs: nba-eval/race3.py) shows training on all earlier seasons beats the
one-season window by 0.0045 log-loss over every regular-season game (95% interval excludes zero) and 0.054 over the
first 15 games, with margin MAE down 0.09. ESPN cannot be re-fetched for eight seasons on every CI run or container
refresh, so completed seasons are frozen in data/history/nba_history.json.gz by scripts/build_history.py. A missing
file means "no history": the retrain then uses the window alone, exactly as before.
"""
from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

import pandas as pd

from nba_predictor import config

logger = logging.getLogger(__name__)

HISTORY_RELATIVE = Path("data") / "history" / "nba_history.json.gz"
# Everything the props need (rolling means of points/rebounds/assists/threes/minutes) and the availability features use.
PLAYER_COLS = [
    "game_id", "game_date", "team", "player_id", "player_name", "position", "minutes", "points", "rebounds", "assists", "fg3m",
    "fga", "home_team", "away_team", "opponent",       # the matchup/usage features of the props model need these
]


def load_history(before: str, path: Path | None = None) -> tuple[list[dict], pd.DataFrame]:
    """(games, player rows) from the frozen file with game_date < `before` (the ingest window start)."""
    # Resolved per call, not at import, so a test or deployment that moves PROJECT_ROOT moves the history with it.
    path = path or config.PROJECT_ROOT / HISTORY_RELATIVE
    if not path.exists():
        logger.warning("no frozen history at %s: training on the ingest window alone", path)
        return [], pd.DataFrame(columns=PLAYER_COLS)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    games = [g for g in data["games"] if g["game_date"] < before]
    players = pd.DataFrame(data["players"], columns=PLAYER_COLS)
    return games, players[players["game_date"] < before].reset_index(drop=True)


def merge_history(games: list[dict], players: pd.DataFrame, before: str, path: Path | None = None) -> tuple[list[dict], pd.DataFrame]:
    """The window's games and player rows with the frozen earlier seasons in front. The live copy of a game wins."""
    hist_games, hist_players = load_history(before, path)
    if not hist_games:
        return games, players
    live_ids = {g["game_id"] for g in games}
    hist_games = [g for g in hist_games if g["game_id"] not in live_ids]
    hist_players = hist_players[~hist_players["game_id"].isin(live_ids)]
    return hist_games + games, pd.concat([hist_players, players], ignore_index=True)
