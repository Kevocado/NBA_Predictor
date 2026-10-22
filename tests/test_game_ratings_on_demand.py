"""On-demand ratings for a finished game whose window was never warmed.

The warmer keeps only a few played dates, so an older finished game's ratings
are computed on demand from box scores in the cache volume, cached on disk keyed
by (game_id, schedule state), and never block a request thread.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nba_predictor.api import facts as facts_mod


def _state() -> tuple[int, int]:
    return (123456789, 100)


def _game(game_id="g-old", game_date="2026-01-15"):
    return {
        "game_id": game_id,
        "game_date": game_date,
        "home_team": "T29",
        "away_team": "T00",
        "completed": True,
        "home_pts": 110,
        "away_pts": 100,
    }


def _clear_memory(monkeypatch):
    monkeypatch.setattr(facts_mod, "_RATINGS_MEMORY", {})


def test_cold_ratings_return_none_without_blocking(monkeypatch):
    """An old game whose window was never warmed: cold, no table, no block.

    The request thread must not wait for the compute; it returns None
    immediately and the warm happens off the request thread.
    """
    _clear_memory(monkeypatch)
    warmed = {"called": False}
    monkeypatch.setattr(facts_mod, "_box_score_history", lambda as_of: pd.DataFrame())
    monkeypatch.setattr(
        facts_mod, "_warm_ratings_for_game", lambda game, state: warmed.__setitem__("called", True),
    )

    payload = facts_mod.ratings_for_game(_game(), _state())

    assert payload is None, "a cold cache serves no table, it does not block"
    assert warmed["called"], "the warm happens off the request thread"


def test_disk_cache_serves_ratings_without_re_warming(monkeypatch, tmp_path):
    """A disk-cached payload (keyed by game_id + schedule state) is served and
    never re-warmed on a request thread."""
    _clear_memory(monkeypatch)
    disk_dir = tmp_path / "ratings"
    monkeypatch.setattr(facts_mod, "_ratings_disk_dir", lambda: disk_dir)
    disk_dir.mkdir(parents=True, exist_ok=True)

    game = _game()
    state = _state()
    payload = {
        "as_of": game["game_date"],
        "n_teams": 2,
        "caption": "Last 15 games each.",
        "metrics": [],
        "teams": {"T29": {}, "T00": {}},
    }
    # As the off-thread warm would have written it.
    facts_mod._write_ratings_disk(game["game_id"], state, payload)

    warmed = {"called": False}
    monkeypatch.setattr(facts_mod, "_box_score_history", lambda as_of: pd.DataFrame())
    monkeypatch.setattr(
        facts_mod, "_warm_ratings_for_game", lambda game, state: warmed.__setitem__("called", True),
    )

    out = facts_mod.ratings_for_game(game, state)

    assert out == payload, "the disk-cached payload is served"
    assert not warmed["called"], "a disk hit does not warm again"


def test_disk_cache_keyed_by_game_and_schedule_state(monkeypatch, tmp_path):
    """The same game under a different schedule state is a different cache entry:
    a schedule change invalidates the old rating rather than serving it stale."""
    _clear_memory(monkeypatch)
    disk_dir = tmp_path / "ratings"
    monkeypatch.setattr(facts_mod, "_ratings_disk_dir", lambda: disk_dir)
    disk_dir.mkdir(parents=True, exist_ok=True)

    game = _game()
    payload_a = {"as_of": game["game_date"], "n_teams": 2, "caption": "a.", "metrics": [], "teams": {}}
    payload_b = {"as_of": game["game_date"], "n_teams": 2, "caption": "b.", "metrics": [], "teams": {}}

    facts_mod._write_ratings_disk(game["game_id"], _state(), payload_a)
    facts_mod._write_ratings_disk(game["game_id"], (999, 999), payload_b)

    monkeypatch.setattr(facts_mod, "_box_score_history", lambda as_of: pd.DataFrame())
    monkeypatch.setattr(facts_mod, "_warm_ratings_for_game", lambda game, state: None)

    assert facts_mod.ratings_for_game(game, _state()) == payload_a
    assert facts_mod.ratings_for_game(game, (999, 999)) == payload_b


def test_compute_reads_box_files_not_the_network(monkeypatch, tmp_path):
    """Box scores come from cached files on disk; ESPN is never called.

    A game two months old gets its ratings from the cache volume, not a live
    ESPN call on any thread.
    """
    espn_mod = __import__("nba_predictor.data.espn", fromlist=["_fetch_json", "_load_cache", "ESPN_CACHE_DIR"])
    cache_dir = tmp_path / "espn"
    cache_dir.mkdir(parents=True, exist_ok=True)
    box = {
        "T29": {"fgm": 40, "fga": 88, "fg3m": 12, "ftm": 10, "fta": 20, "oreb": 9, "dreb": 32, "tov": 11},
        "T00": {"fgm": 34, "fga": 89, "fg3m": 9, "ftm": 8, "fta": 15, "oreb": 8, "dreb": 27, "tov": 13},
    }
    (cache_dir / "boxscore_g1.json").write_text(json.dumps(box))
    monkeypatch.setattr(espn_mod, "ESPN_CACHE_DIR", cache_dir)

    network = {"called": False}
    monkeypatch.setattr(espn_mod, "_fetch_json", lambda *a, **k: network.__setitem__("called", True) or (_ for _ in ()).throw(AssertionError("ESPN was called")))

    # A schedule with one completed game g1 whose box score is cached.
    monkeypatch.setattr(facts_mod, "_schedule", lambda: [_game(game_id="g1", game_date="2026-01-14")])
    # _build path: lookback window, then select_window keeps it (each team's last game).
    enriched = facts_mod._enrich_from_cache([_game(game_id="g1", game_date="2026-01-14")])

    assert not network["called"], "no ESPN call: the box score came from a cached file"
    assert enriched[0].get("home_fgm") == 40, "the cached box score was read into the frame"
    assert enriched[0].get("away_fgm") == 34
