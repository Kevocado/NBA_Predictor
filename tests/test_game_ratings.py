"""The ratings a reader sees on a game page: a value and a league rank.

The load-bearing property here is that a rank means something. A rank out of
fewer than 30 teams is a fabricated position -- the league is 30 teams, so a
"rank 4 of 14" is not a place in the league -- and a value with no rank is a
number with nothing to compare it to.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nba_predictor.api import facts as facts_mod
from nba_predictor.signals.team_matchups import RATING_METRICS, league_ratings
from tests.test_four_factors_duel import DATES, TEAMS, _schedule


def _league(flip: bool = False) -> pd.DataFrame:
    """A 30-team league in the pipeline's own frame shape."""
    frame = _schedule(DATES).copy()
    for side in ("home", "away"):
        frame[f"{side}_fg3a"] = 25.0
    if flip:
        for side in ("home", "away"):
            frame[f"{side}_fgm"] = 100 - frame[f"{side}_fgm"]
    return frame


def test_the_ratings_cover_every_metric_the_reader_is_shown():
    ratings = league_ratings(_league(), "2026-03-18")

    assert len(ratings) == 30, "the league is 30 teams"
    for key, _label, _unit, _higher in RATING_METRICS:
        assert key in ratings.columns, f"{key} is not in the ratings"
        assert f"{key}_rank" in ratings.columns, f"{key} has no rank"
        ranks = ratings[f"{key}_rank"].dropna()
        assert len(ranks) == 30, f"{key} is ranked for {len(ranks)} of 30 teams"
        assert ranks.min() >= 1 and ranks.max() <= 30


def test_a_defence_is_ranked_by_what_it_allows():
    """The best defence is the one that allows the least, so a team that
    concedes the fewest points ranks first on Defence."""
    ratings = league_ratings(_league(), "2026-03-18")

    allowed = ratings["drtg"]
    best_defence = allowed.idxmin()
    worst_defence = allowed.idxmax()

    assert ratings.loc[best_defence, "drtg_rank"] == 1
    assert ratings.loc[worst_defence, "drtg_rank"] == ratings["drtg_rank"].max(), (
        "the team that concedes the most is not the worst-placed defence"
    )


def test_the_ranks_follow_the_league_when_it_flips():
    best = league_ratings(_league(), "2026-03-18")
    flipped = league_ratings(_league(flip=True), "2026-03-18")

    assert best.loc["T29", "efg_pct_rank"] == 1
    assert flipped.loc["T29", "efg_pct_rank"] == 30


def test_the_payload_carries_both_teams_and_the_window_it_read(monkeypatch):
    """The page shows two teams against the league, and says what the window
    was -- a reader who does not know the window is reading a number they
    cannot check."""
    monkeypatch.setattr(facts_mod, "_box_score_history", lambda as_of: _league())

    payload = facts_mod.ratings_payload("2026-03-18", ["T29", "T00"])

    assert payload is not None
    assert payload["n_teams"] == 30
    assert set(payload["teams"]) == {"T29", "T00"}
    assert len(payload["metrics"]) == len(RATING_METRICS)
    assert payload["caption"].startswith("Last 15 games each")
    for team, values in payload["teams"].items():
        for key, _label, _unit, _higher in RATING_METRICS:
            assert set(values[key]) == {"value", "rank"}, f"{team}/{key} is not a value and a rank"
            assert 1 <= values[key]["rank"] <= 30


def test_the_payload_is_none_when_the_league_is_not_ranked(monkeypatch):
    """No ratings is not a section full of zeros."""
    monkeypatch.setattr(facts_mod, "_box_score_history", lambda as_of: pd.DataFrame())

    assert facts_mod.ratings_payload("2026-03-18", ["T29", "T00"]) is None


def test_the_caption_says_when_the_window_reaches_into_last_season(monkeypatch):
    """Early in a season a team has not played 15 games, so the window reaches
    into last season -- and a rating read off last year's form has to say so."""
    frame = _league()
    frame["game_date"] = "2026-03-01"
    monkeypatch.setattr(facts_mod, "_box_score_history", lambda as_of: frame)

    payload = facts_mod.ratings_payload("2026-10-21", ["T29", "T00"])

    assert "last season" in payload["caption"], payload["caption"]


def test_the_caption_does_not_mention_last_season_mid_season(monkeypatch):
    frame = _league()
    frame["game_date"] = "2026-11-15"
    monkeypatch.setattr(facts_mod, "_box_score_history", lambda as_of: frame)

    payload = facts_mod.ratings_payload("2026-12-01", ["T29", "T00"])

    assert "last season" not in payload["caption"], payload["caption"]


def test_a_missing_figure_is_null_and_not_zero(monkeypatch):
    """A team that shot no threes has no 3P%. Zero would read as a team that
    shot none and made none."""
    frame = _league()
    frame["home_fg3a"] = 0.0
    frame["away_fg3a"] = 0.0
    monkeypatch.setattr(facts_mod, "_box_score_history", lambda as_of: frame)

    payload = facts_mod.ratings_payload("2026-03-18", ["T29", "T00"])

    assert payload["teams"]["T29"]["fg3_pct"]["value"] is None
    assert payload["teams"]["T29"]["fg3_pct"]["rank"] is None


def test_the_game_detail_serves_the_ratings(tmp_path, monkeypatch):
    """The page reads them from the game's own payload."""
    import json

    from fastapi.testclient import TestClient

    from nba_predictor.api import deps
    from nba_predictor.api.app import app
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(
        json.dumps([{"game_id": "g1", "game_date": "2026-03-18", "home_team": "T29", "away_team": "T00"}])
    )
    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path
    app.dependency_overrides[deps.get_injury_report_best_effort] = lambda: []
    monkeypatch.setattr(facts_mod, "_box_score_history", lambda as_of: _league())

    client = TestClient(app)
    body = client.get("/games/g1").json()
    app.dependency_overrides.clear()

    assert body["ratings"] is not None
    assert body["ratings"]["n_teams"] == 30
    assert set(body["ratings"]["teams"]) == {"T29", "T00"}


def test_a_finished_game_serves_the_ratings_as_they_stood_before_it(monkeypatch):
    """A finished game shows the form each side took into it.

    The window is built from the games before the game's own date, so the game
    is not in its own window, and the caption says which is which.
    """
    monkeypatch.setattr(facts_mod, "_box_score_history", lambda as_of: _league())

    payload = facts_mod.ratings_payload("2026-10-20", ["T00", "T29"], before_game=True)

    assert payload is not None
    assert payload["caption"].startswith("Before the game.")
    assert "Last 15 games each" in payload["caption"]


def test_a_game_still_to_come_does_not_say_before_the_game(monkeypatch):
    monkeypatch.setattr(facts_mod, "_box_score_history", lambda as_of: _league())

    payload = facts_mod.ratings_payload("2026-10-21", ["T00", "T29"])

    assert payload is not None
    assert not payload["caption"].startswith("Before the game.")
