"""The Players page's extra data: each player's next game, and their last ten.

The load-bearing property here is that a missing projection is `None` and
never zero: a player with no stored projection has no projection, and zero
would read as "the model expects nothing from him".
"""
from __future__ import annotations

import pytest

from nba_predictor.api.player_details import next_game_projections, player_detail
from nba_predictor.tracking import store


def _game(game_id, date_, home, away, completed=True, home_pts=None, away_pts=None):
    return {
        "game_id": game_id,
        "game_date": date_,
        "home_team": home,
        "away_team": away,
        "completed": completed,
        "home_pts": home_pts,
        "away_pts": away_pts,
        "tip_off": f"{date_}T19:00Z",
    }


SCHEDULE = [
    _game("g1", "2026-10-10", "BOS", "MIA", True, 110, 100),
    _game("g2", "2026-10-12", "MIA", "BOS", True, 95, 105),
    _game("g3", "2026-10-21", "BOS", "MIA", False),
]


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "tracking.db"
    store.init_db(path)
    return path


def test_each_player_gets_their_teams_next_game(db):
    store.insert_player_prediction(
        db, game_id="g3", player_id="p1", stat="points", predicted_value=24.0,
        created_at="2026-10-20T12:00:00",
    )

    out = next_game_projections(SCHEDULE, db, {"p1": "BOS"}, "2026-10-20")

    assert out["p1"]["game_id"] == "g3"
    assert out["p1"]["opponent"] == "MIA"
    assert out["p1"]["home"] is True
    assert out["p1"]["projections"]["points"] == 24.0


def test_a_projection_that_is_not_there_is_null_and_not_zero(db):
    out = next_game_projections(SCHEDULE, db, {"p1": "BOS"}, "2026-10-20")

    assert out["p1"]["projections"]["points"] is None
    assert out["p1"]["projections"]["assists"] is None


def test_a_player_with_no_next_game_is_absent(db):
    out = next_game_projections(SCHEDULE[:2], db, {"p1": "BOS"}, "2026-10-20")

    assert "p1" not in out


def test_the_newest_stored_row_wins(db):
    """The store keeps every snapshot; the newest is the one the site shows."""
    store.insert_player_prediction(
        db, game_id="g3", player_id="p1", stat="points", predicted_value=24.0,
        created_at="2026-10-20T12:00:00",
    )
    store.insert_player_prediction(
        db, game_id="g3", player_id="p1", stat="points", predicted_value=27.5,
        created_at="2026-10-20T18:00:00",
    )

    out = next_game_projections(SCHEDULE, db, {"p1": "BOS"}, "2026-10-20")

    assert out["p1"]["projections"]["points"] == 27.5


def test_the_drawer_shows_the_last_ten_games_with_the_actual_and_the_projection(db):
    store.insert_player_outcome(
        db, game_id="g1", player_id="p1", stat="points", actual_value=30.0,
        recorded_at="2026-10-10T23:00:00",
    )
    store.insert_player_prediction(
        db, game_id="g1", player_id="p1", stat="points", predicted_value=22.0,
        created_at="2026-10-09T12:00:00",
    )

    detail = player_detail(SCHEDULE, db, "p1", "BOS", [], "2026-10-20")

    assert len(detail["games"]) == 1
    game = detail["games"][0]
    assert game["opponent"] == "MIA"
    assert game["score"] == "110–100"
    assert game["actual"]["points"] == 30.0
    assert game["projection"]["points"] == 22.0


def test_the_drawer_names_the_players_status(db):
    detail = player_detail(
        SCHEDULE, db, "p1", "BOS",
        [{"player_id": "p1", "status": "Out", "source": "ESPN injury report", "dated": "2026-10-19"}],
        "2026-10-20",
    )

    assert detail["status"]["status"] == "Out"
    assert detail["status"]["source"] == "ESPN injury report"


def test_a_player_with_no_status_has_none(db):
    detail = player_detail(SCHEDULE, db, "p1", "BOS", [], "2026-10-20")

    assert detail["status"] is None
