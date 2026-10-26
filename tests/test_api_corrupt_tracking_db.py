"""A broken tracking DB must not take down a read-only page.

On 2026-10-26 the tracking DB was corrupted by a mid-commit I/O error. Every
read raised `sqlite3.DatabaseError: database disk image is malformed`, and
because `_prediction_out` did not catch it, `/games/week` answered 500 for every
week. The whole NBA games page fell back to "We couldn't load this week's games.
Check your connection and try again." -- a data problem rendered as a network
problem, with no games at all.

A prediction is an OPTIONAL field: the schema is `prediction: Prediction | null`
and the frontend already draws a game without one. Serving the game with no pick
is the truthful degradation. Serving a 500 says the schedule is unavailable,
which was never true.
"""
import pytest


CORRUPT = b"this is not a sqlite database\n" * 4096


def _client_with_corrupt_db(tmp_path):
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import app
    from nba_predictor.api import deps

    db_path = tmp_path / "tracking.db"
    db_path.write_bytes(CORRUPT)

    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(
        '[{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}]'
    )

    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path
    app.dependency_overrides[deps.get_injury_report] = lambda: []
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    from nba_predictor.api.app import app

    yield
    app.dependency_overrides.clear()


def test_the_fixture_really_is_unreadable(tmp_path):
    """Guard the guard: a test of graceful degradation is vacuous if the DB it
    is handed happens to open fine."""
    import sqlite3

    path = tmp_path / "broken.db"
    path.write_bytes(CORRUPT)
    with pytest.raises(sqlite3.DatabaseError):
        sqlite3.connect(f"file:{path}", uri=True).execute(
            "select count(*) from predictions"
        ).fetchall()


def test_week_list_still_serves_games_without_the_broken_db(tmp_path):
    client = _client_with_corrupt_db(tmp_path)

    response = client.get("/games/week", params={"start": "2026-10-26"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body) == 1
    assert body[0]["game_id"] == "g1"
    assert body[0]["prediction"] is None, (
        "a game carried a prediction read from a database that cannot be read"
    )


def test_a_single_day_list_also_degrades(tmp_path):
    client = _client_with_corrupt_db(tmp_path)

    response = client.get("/games", params={"date": "2026-11-01"})

    assert response.status_code == 200, response.text
    assert [g["game_id"] for g in response.json()] == ["g1"]


def test_the_game_detail_degrades_too(tmp_path):
    client = _client_with_corrupt_db(tmp_path)

    response = client.get("/games/g1")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["game_id"] == "g1"
    assert body["prediction"] is None
    assert body["markets"] == []
