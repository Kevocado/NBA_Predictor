"""The Teams page's data: the league's ratings, every team's form, its next game.

The load-bearing property here is that the form is honest about what has
happened: a game with no score is not a result, and a game that has not been
played is not in the next-game column.
"""
from __future__ import annotations

import pytest

from nba_predictor.api.team_overview import (
    DETAIL_GAMES,
    FORM_GAMES,
    form_for_team,
    last_games_for_team,
    next_game_for_team,
)


def _game(game_id, date_, home, away, home_pts=None, away_pts=None, completed=True, tip_off=None):
    return {
        "game_id": game_id,
        "game_date": date_,
        "home_team": home,
        "away_team": away,
        "home_pts": home_pts,
        "away_pts": away_pts,
        "completed": completed,
        "tip_off": tip_off,
    }


SCHEDULE = [
    _game("g1", "2026-10-10", "BOS", "MIA", 110, 100),
    _game("g2", "2026-10-11", "MIA", "BOS", 95, 105),
    _game("g3", "2026-10-12", "BOS", "MIA", 90, 100),
    _game("g4", "2026-10-13", "MIA", "BOS", 120, 118),
    _game("g5", "2026-10-21", "BOS", "MIA", None, None, completed=False, tip_off="2026-10-21T19:00Z"),
]


def test_the_form_is_the_last_ten_results_oldest_first():
    form = form_for_team(SCHEDULE, "BOS")

    assert form == ["W", "W", "L", "L"]
    assert len(form) <= FORM_GAMES


def test_an_unplayed_game_is_not_a_result():
    """A game that has not happened contributes nothing to the form."""
    form = form_for_team(SCHEDULE, "BOS", before="2026-10-21")

    assert "L" not in form[-1:] or form[-1] == "L"
    assert len(form) == 4


def test_the_last_five_games_carry_the_score_and_the_result():
    games = last_games_for_team(SCHEDULE, "BOS")

    assert len(games) <= DETAIL_GAMES
    assert games[-1]["opponent"] == "MIA"
    assert games[-1]["score"] == "118–120"      # BOS (away) scored 118: his own score first, beside the L
    assert games[-1]["result"] == "L"
    assert games[-1]["home"] is False


def test_the_next_game_is_the_first_one_still_to_come(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    game = next_game_for_team(SCHEDULE, "BOS", db_path=db_path, today="2026-10-20")

    assert game is not None
    assert game["game_id"] == "g5"
    assert game["opponent"] == "MIA"
    assert game["home"] is True


def test_a_team_with_no_upcoming_game_has_no_next_game(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    assert next_game_for_team(SCHEDULE[:4], "BOS", db_path=db_path, today="2026-10-20") is None


def test_a_team_with_no_pick_has_no_pick_in_its_next_game(tmp_path):
    """No pre-tip row means the model has not said anything about the game yet."""
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    # A row made AFTER tip-off is not a pre-tip pick.
    store.insert_prediction(
        db_path, game_id="g5", created_at="2026-10-21T20:00:00+00:00", model_version="v1",
        home_win_prob=0.6, predicted_margin=3.0, predicted_total=220.0,
    )

    game = next_game_for_team(SCHEDULE, "BOS", db_path=db_path, today="2026-10-20")

    assert game is not None
    assert game["pick"] is None
    assert game["pick_probability"] is None


def test_a_pre_tip_pick_is_carried_with_its_probability(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_prediction(
        db_path, game_id="g5", created_at="2026-10-20T12:00:00+00:00", model_version="v1",
        home_win_prob=0.62, predicted_margin=3.0, predicted_total=220.0,
    )

    game = next_game_for_team(SCHEDULE, "BOS", db_path=db_path, today="2026-10-20")

    assert game["pick"] == "BOS"
    assert game["pick_probability"] == pytest.approx(0.62)
