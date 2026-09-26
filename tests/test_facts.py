"""Tests for the read-only NBA /facts bundle the match explainer consumes.

Offline throughout: the schedule, the SQLite path and the store are all
injected, so nothing reaches balldontlie, the NBA API or the odds feed.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, model_validator

from nba_predictor.api import facts as facts_mod
from nba_predictor.api.app import create_app

# The contract, copied from predictor-hub/services/explainer/explainer/facts.py.
class Market(BaseModel):
    market: str
    model_config = ConfigDict(extra="allow")


class Facts(BaseModel):
    sport: Literal["pl", "f1", "nfl", "cfb", "nba"]
    id: str
    title: str
    starts_at: str
    status: Literal["upcoming", "live", "final"]
    pick_timing: Literal["pre_kickoff", "rebuilt", "none"]
    pick: dict | None = None
    markets: list[Market] = []
    drivers: list[dict] = []
    context: dict = {}
    players: list[dict] = []
    record: dict | None = None
    result: dict | None = None

    @model_validator(mode="after")
    def _rebuilt_never_won(self) -> "Facts":
        if self.pick_timing == "rebuilt" and self.result and "pick_won" in self.result:
            raise ValueError("a rebuilt pick cannot carry result.pick_won")
        return self


NOW = datetime(2026, 1, 15, 18, 0, tzinfo=timezone.utc)
GAME_ID = "401812480"


def _game(**over):
    game = {
        "game_id": GAME_ID,
        "game_date": "2026-01-16",
        "tip_off": "2026-01-16T00:30:00Z",
        "home_team": "BOS",
        "away_team": "MIA",
        "completed": False,
        "home_pts": None,
        "away_pts": None,
    }
    game.update(over)
    return game


def _prediction_row(**over):
    row = {
        "home_win_prob": 0.62,
        "predicted_margin": 3.46,
        "predicted_total": 224.5,
        "created_at": "2026-01-15T12:00:00Z",  # before tip-off
    }
    row.update(over)
    return row


def _market_row(**over):
    row = {
        "market": "spread",
        "selection": "BOS",
        "model_probability": 0.55,
        "market_probability": 0.5,
        "edge": 0.05,
        "bookmaker": "draftkings",
        "american_odds": -110,
        "point": -3.5,
        "created_at": "2026-01-15T12:00:00Z",
    }
    row.update(over)
    return row


def _player_row(**over):
    row = {
        "player_id": "p1",
        "player_name": "Jayson Tatum",
        "stat": "points",
        "predicted_value": 28.4,
        "actual_value": None,
        "rebuilt": False,
    }
    row.update(over)
    return row


@pytest.fixture
def api(monkeypatch):
    schedule = [_game()]
    monkeypatch.setattr(facts_mod, "_now", lambda: NOW)
    monkeypatch.setattr(facts_mod, "_schedule", lambda: schedule)
    monkeypatch.setattr(facts_mod, "_db_path", lambda: Path("/nonexistent/test.sqlite"))
    monkeypatch.setattr(facts_mod, "_predictions", lambda game_id: [_prediction_row()])
    monkeypatch.setattr(facts_mod, "_market_rows", lambda game_id: [_market_row()])
    monkeypatch.setattr(facts_mod, "_player_rows", lambda game_id: [
        _player_row(), _player_row(player_id="p2", player_name="Bam Adebayo", stat="rebounds", predicted_value=11.2),
    ])
    monkeypatch.setattr(facts_mod, "_track_record", lambda: {"total_predictions": 88, "correct_predictions": 55, "hit_rate": 0.625, "n_rebuilt": 4})
    return TestClient(create_app())


# --- the contract -------------------------------------------------------

def test_bundle_validates_against_the_contract(api):
    body = api.get(f"/facts/{GAME_ID}").json()

    Facts(**body)
    assert body["sport"] == "nba"
    assert body["id"] == GAME_ID
    assert body["title"] == "MIA at BOS"
    assert body["starts_at"] == "2026-01-16T00:30:00Z"
    assert body["status"] == "upcoming"
    assert body["pick_timing"] == "pre_kickoff"
    assert body["pick"] == {"label": "BOS", "prob": 0.62}


def test_moneyline_spread_and_total_come_from_the_prediction(api):
    body = api.get(f"/facts/{GAME_ID}").json()
    by_market = {m["market"]: m for m in body["markets"]}

    assert by_market["moneyline"]["model"] == {"BOS": pytest.approx(0.62), "MIA": pytest.approx(0.38)}
    # marginLine semantics, ported from frontend/src/lib/pick.ts: the team
    # with the margin must be the pick, and under half a point is a Toss-up.
    assert by_market["spread"]["line"] == "BOS by 3.5"
    assert by_market["total"]["model_total"] == pytest.approx(224.5)


def test_spread_reads_toss_up_under_half_a_point(api, monkeypatch):
    monkeypatch.setattr(facts_mod, "_predictions", lambda game_id: [_prediction_row(predicted_margin=0.3)])

    body = api.get(f"/facts/{GAME_ID}").json()
    spread = next(m for m in body["markets"] if m["market"] == "spread")

    assert spread["line"] == "Toss-up"


def test_spread_reads_toss_up_when_the_margin_points_at_the_other_team(api, monkeypatch):
    # The win and margin models disagree: the margin favours MIA, the pick is BOS.
    monkeypatch.setattr(facts_mod, "_predictions", lambda game_id: [_prediction_row(predicted_margin=-4.0)])

    body = api.get(f"/facts/{GAME_ID}").json()
    spread = next(m for m in body["markets"] if m["market"] == "spread")

    assert spread["line"] == "Toss-up"


def test_market_lines_use_pre_tip_rows_only(api, monkeypatch):
    monkeypatch.setattr(facts_mod, "_market_rows", lambda game_id: [
        _market_row(created_at="2026-01-16T02:00:00Z", point=-9.5),  # after tip-off
        _market_row(created_at="2026-01-15T12:00:00Z", point=-3.5),
    ])

    body = api.get(f"/facts/{GAME_ID}").json()
    spread = next(m for m in body["markets"] if m["market"] == "spread")

    # `line` is the model's own margin statement (marginLine semantics);
    # `market_line` is the newest PRE-tip row's own line. The post-tip -9.5
    # is ignored.
    assert spread["line"] == "BOS by 3.5"
    assert spread["market_line"] == "BOS -3.5"
    assert "-9.5" not in str(spread)


def test_players_exclude_rebuilt_projections(api, monkeypatch):
    monkeypatch.setattr(facts_mod, "_player_rows", lambda game_id: [
        _player_row(player_id="p1", player_name="Kept", stat="points", predicted_value=28.4),
        _player_row(player_id="p2", player_name="Dropped", stat="points", predicted_value=31.0, rebuilt=True),
    ])

    body = api.get(f"/facts/{GAME_ID}").json()

    assert [p["name"] for p in body["players"]] == ["Kept"]


def test_players_are_the_top_three_by_projection(api, monkeypatch):
    monkeypatch.setattr(facts_mod, "_player_rows", lambda game_id: [
        _player_row(player_id="p1", player_name="A", stat="points", predicted_value=10.0),
        _player_row(player_id="p2", player_name="B", stat="points", predicted_value=30.0),
        _player_row(player_id="p3", player_name="C", stat="rebounds", predicted_value=20.0),
        _player_row(player_id="p4", player_name="D", stat="assists", predicted_value=5.0),
    ])

    body = api.get(f"/facts/{GAME_ID}").json()

    assert [p["name"] for p in body["players"]] == ["B", "C", "A"]


def test_record_uses_the_game_outcome_row(api):
    body = api.get(f"/facts/{GAME_ID}").json()

    assert body["record"] == {"label": "Picks made before tip-off", "hits": 55, "settled": 88}


def test_context_is_omitted_when_the_schedule_has_no_rest_fields(api):
    body = api.get(f"/facts/{GAME_ID}").json()

    # Nothing is invented: no rest computation when the schedule lacks it.
    assert "rest" not in body["context"]


# --- pick_timing: the three cases ---------------------------------------

def test_pick_timing_is_pre_kickoff_for_a_pre_tip_row(api):
    body = api.get(f"/facts/{GAME_ID}").json()

    assert body["pick_timing"] == "pre_kickoff"
    assert body["pick"]["label"] == "BOS"


def test_pick_timing_is_rebuilt_when_only_a_post_tip_row_exists(api, monkeypatch):
    monkeypatch.setattr(facts_mod, "_predictions", lambda game_id: [
        _prediction_row(created_at="2026-01-16T02:00:00Z"),  # after tip-off
    ])

    body = api.get(f"/facts/{GAME_ID}").json()

    assert body["pick_timing"] == "rebuilt"
    assert body["pick"] is not None


def test_pick_timing_is_none_when_there_is_no_prediction(api, monkeypatch):
    monkeypatch.setattr(facts_mod, "_predictions", lambda game_id: [])

    body = api.get(f"/facts/{GAME_ID}").json()

    assert body["pick_timing"] == "none"
    assert body["pick"] is None


# --- started games: only the pre-tip pick --------------------------------

def test_started_game_judges_the_pre_tip_row_not_the_post_tip_one(api, monkeypatch):
    started = _game(tip_off="2026-01-14T00:30:00Z", game_date="2026-01-14")
    monkeypatch.setattr(facts_mod, "_schedule", lambda: [started])
    # A backtest row made after tip-off flips the favourite; the pre-tip row
    # is the only pick that may be judged.
    monkeypatch.setattr(facts_mod, "_predictions", lambda game_id: [
        _prediction_row(home_win_prob=0.62, created_at="2026-01-13T12:00:00Z"),
        _prediction_row(home_win_prob=0.40, created_at="2026-01-14T02:00:00Z"),
    ])

    body = api.get(f"/facts/{GAME_ID}").json()

    assert body["status"] == "live"
    assert body["pick"] == {"label": "BOS", "prob": 0.62}
    assert body["pick_timing"] == "pre_kickoff"


def test_started_game_without_a_pre_tip_row_has_no_pick(api, monkeypatch):
    started = _game(tip_off="2026-01-14T00:30:00Z", game_date="2026-01-14")
    monkeypatch.setattr(facts_mod, "_schedule", lambda: [started])
    monkeypatch.setattr(facts_mod, "_predictions", lambda game_id: [
        _prediction_row(home_win_prob=0.40, created_at="2026-01-14T02:00:00Z"),
    ])

    body = api.get(f"/facts/{GAME_ID}").json()

    # A post-tip-only row is never the pre-start pick, so a started game with
    # no pre-tip row has no pick at all: null, timing 'none', no markets.
    assert body["status"] == "live"
    assert body["pick"] is None
    assert body["pick_timing"] == "none"
    assert body["markets"] == []


# --- finals -------------------------------------------------------------

def test_final_has_a_score_and_pick_won_when_pre_kickoff(api, monkeypatch):
    final = _game(
        game_date="2026-01-14", tip_off="2026-01-14T00:30:00Z",
        completed=True, home_pts=112, away_pts=104,
    )
    monkeypatch.setattr(facts_mod, "_schedule", lambda: [final])
    monkeypatch.setattr(facts_mod, "_predictions", lambda game_id: [
        _prediction_row(home_win_prob=0.62, created_at="2026-01-13T12:00:00Z"),
    ])

    body = api.get(f"/facts/{GAME_ID}").json()

    assert body["status"] == "final"
    assert body["result"]["score"] == "BOS 112-104"
    assert body["result"]["pick_won"] is True
    Facts(**body)


def test_final_omits_pick_won_when_the_pick_was_made_after_tip(api, monkeypatch):
    final = _game(
        game_date="2026-01-14", tip_off="2026-01-14T00:30:00Z",
        completed=True, home_pts=104, away_pts=112,
    )
    monkeypatch.setattr(facts_mod, "_schedule", lambda: [final])
    monkeypatch.setattr(facts_mod, "_predictions", lambda game_id: [
        _prediction_row(home_win_prob=0.62, created_at="2026-01-14T02:00:00Z"),  # after tip
    ])

    body = api.get(f"/facts/{GAME_ID}").json()

    assert body["status"] == "final"
    assert "score" in body["result"]
    # No pre-tip pick exists, so nothing is judged: no pick_won, whatever the
    # post-tip row happened to say.
    assert body["pick_timing"] == "none"
    assert "pick_won" not in body["result"]
    Facts(**body)


def test_upcoming_game_has_no_result(api):
    body = api.get(f"/facts/{GAME_ID}").json()

    assert body["result"] is None


# --- /facts/upcoming ----------------------------------------------------

def test_upcoming_lists_only_games_inside_the_window(api, monkeypatch):
    soon = _game(game_id="1", game_date="2026-01-15", tip_off=(NOW + timedelta(hours=10)).isoformat().replace("+00:00", "Z"))
    later = _game(game_id="2", game_date="2026-01-20", tip_off=(NOW + timedelta(hours=100)).isoformat().replace("+00:00", "Z"))
    past = _game(game_id="3", game_date="2026-01-10", tip_off=(NOW - timedelta(hours=10)).isoformat().replace("+00:00", "Z"))
    monkeypatch.setattr(facts_mod, "_schedule", lambda: [soon, later, past])

    body = api.get("/facts/upcoming?hours=72").json()

    assert body["ids"] == ["1"]


def test_unknown_game_id_is_404(api):
    assert api.get("/facts/999999999").status_code == 404
