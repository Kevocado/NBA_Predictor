"""On-demand ratings for a finished game whose window was never warmed.

The warmer keeps only a few played dates, so an older finished game's ratings
are computed on demand from box scores in the cache volume, cached on disk keyed
by the game, and never block a request thread.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nba_predictor.api import facts as facts_mod


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
        facts_mod, "_warm_ratings_for_game", lambda game: warmed.__setitem__("called", True),
    )

    payload = facts_mod.ratings_for_game(_game())

    assert payload is None, "a cold cache serves no table, it does not block"
    assert warmed["called"], "the warm happens off the request thread"


def test_disk_cache_serves_ratings_without_re_warming(monkeypatch, tmp_path):
    """A disk-cached payload (keyed by the game) is served and
    never re-warmed on a request thread."""
    _clear_memory(monkeypatch)
    disk_dir = tmp_path / "ratings"
    monkeypatch.setattr(facts_mod, "_ratings_disk_dir", lambda: disk_dir)
    disk_dir.mkdir(parents=True, exist_ok=True)

    game = _game()
    payload = {
        "as_of": game["game_date"],
        "n_teams": 2,
        "caption": "Last 15 games each.",
        "metrics": [],
        "teams": {"T29": {}, "T00": {}},
    }
    # As the off-thread warm would have written it.
    facts_mod._write_ratings_disk(game["game_id"], payload)

    warmed = {"called": False}
    monkeypatch.setattr(facts_mod, "_box_score_history", lambda as_of: pd.DataFrame())
    monkeypatch.setattr(
        facts_mod, "_warm_ratings_for_game", lambda game: warmed.__setitem__("called", True),
    )

    out = facts_mod.ratings_for_game(game)

    assert out == payload, "the disk-cached payload is served"
    assert not warmed["called"], "a disk hit does not warm again"


def test_a_schedule_rewrite_does_not_invalidate_a_finished_games_ratings(monkeypatch, tmp_path):
    """The scores timer rewrites the schedule file every few minutes on a game day; a finished game's pre-game ratings do not
    change with it, so the cache is keyed by the game alone (no recompute, no file left behind per schedule state)."""
    _clear_memory(monkeypatch)
    disk_dir = tmp_path / "ratings"
    monkeypatch.setattr(facts_mod, "_ratings_disk_dir", lambda: disk_dir)
    disk_dir.mkdir(parents=True, exist_ok=True)
    game = _game()
    payload = {"as_of": game["game_date"], "n_teams": 2, "caption": "a.", "metrics": [], "teams": {}}
    facts_mod._write_ratings_disk(game["game_id"], payload)
    monkeypatch.setattr(facts_mod, "_box_score_history", lambda as_of: pd.DataFrame())
    monkeypatch.setattr(facts_mod, "_warm_ratings_for_game", lambda game: pytest.fail("re-warmed"))

    assert facts_mod.ratings_for_game(game) == payload
    assert facts_mod.ratings_for_game(game) == payload
    assert [p.name for p in disk_dir.iterdir()] == [f"{game['game_id']}.json"]


def test_a_burst_of_requests_for_one_cold_game_starts_one_computation(monkeypatch):
    _clear_memory(monkeypatch)
    monkeypatch.setattr(facts_mod, "_RATINGS_INFLIGHT", set())
    monkeypatch.setattr(facts_mod, "_box_score_history", lambda as_of: pd.DataFrame())
    monkeypatch.setattr(facts_mod, "_read_ratings_disk", lambda game_id: None)
    started = []

    class _Thread:
        def __init__(self, target, args, daemon):
            started.append(args)

        def start(self):
            pass

    monkeypatch.setattr(facts_mod.threading, "Thread", _Thread)
    for _ in range(5):
        assert facts_mod.ratings_for_game(_game()) is None
    assert len(started) == 1


def test_a_failure_in_the_ratings_never_takes_the_game_page_down(monkeypatch):
    from nba_predictor.api import routes

    def boom(*args, **kwargs):
        raise RuntimeError("box history exploded")

    monkeypatch.setattr(routes.facts_mod, "ratings_for_game", boom)
    monkeypatch.setattr(routes.facts_mod, "ratings_payload", boom)
    assert routes._game_ratings(_game()) is None
    assert routes._game_ratings({**_game(), "completed": False}) is None


def test_an_upcoming_game_does_not_get_the_before_the_game_wording(monkeypatch):
    from nba_predictor.api import routes

    seen = {}

    def fake(as_of, teams, before_game=False, history=None):
        seen["before_game"] = before_game
        return {}

    monkeypatch.setattr(routes.facts_mod, "ratings_payload", fake)
    routes._game_ratings({**_game(), "completed": False})
    assert seen["before_game"] is False


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
