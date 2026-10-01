"""The props endpoint: an in-sample MAE on every row, and the gate applied.

/games/{id}/players stays a bare JSON array of PlayerPropOut -- the frontend
client already types it that way (frontend/src/api/client.ts getGamePlayers),
so the out entries get their own endpoint rather than a wrapper this backend
PR cannot add without rewriting the frontend it is not allowed to touch.

Two invariants asserted here:

  * no probability is served anywhere on the row. NBA's model is an
    XGBRegressor returning a raw point total; there is no calibrated
    probability to show, and the phase forbids inventing one. Every row is a
    projection.
  * an out player's row is ABSENT from the ranking, not flagged in place, and
    appears exactly once below it with a source and a date.
"""

import json

import pytest
from fastapi.testclient import TestClient

from nba_predictor.api.app import app
from nba_predictor.api import deps
from nba_predictor.tracking import store


SCHEDULE = [
    {
        "game_id": "g1",
        "game_date": "2026-02-16",
        "home_team": "BOS",
        "away_team": "MIA",
        "tip_off": "2026-02-16T00:30Z",
    },
    {
        "game_id": "g2",
        "game_date": "2026-11-01",
        "home_team": "BOS",
        "away_team": "MIA",
        "tip_off": "2026-11-01T00:30Z",
    },
]

PRE_TIP = "2026-02-16T00:00:00+00:00"  # before g1's tip-off
POST_TIP = "2026-09-20T08:00:00+00:00"  # after it: a backtest row
UPCOMING = "2026-11-01T00:00:00+00:00"  # before g2's tip-off


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(json.dumps(SCHEDULE))

    hub_dir = tmp_path / "data" / "cache" / "hub"
    hub_dir.mkdir(parents=True)
    (hub_dir / "players.json").write_text(
        json.dumps(
            [
                {"player_id": "5105571", "player_name": "Jayson Tatum"},
                {"player_id": "3934672", "player_name": "Nikola Jokic"},
                {"player_id": "203999", "player_name": "Someone Healthy"},
            ]
        )
    )

    from nba_predictor import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path
    # The gate's feed is a dependency, so no test here can reach the network.
    app.dependency_overrides[deps.get_injury_report] = lambda: []

    yield TestClient(app)
    app.dependency_overrides.clear()


def _injuries(*rows):
    app.dependency_overrides[deps.get_injury_report] = lambda: list(rows)


def _seed_resolved(db_path):
    """g1, resolved before tip-off on two players.

    points: 5105571 off by 4 and 3934672 off by 2 -> mean 3.
    threes: one resolved row, off by 1.
    rebounds and assists: predicted, never resolved.
    """
    for player_id, predicted, actual in (("5105571", 20.0, 24.0), ("3934672", 10.0, 12.0)):
        store.insert_player_prediction(
            db_path, game_id="g1", player_id=player_id, stat="points",
            predicted_value=predicted, created_at=PRE_TIP,
        )
        store.insert_player_outcome(
            db_path, game_id="g1", player_id=player_id, stat="points",
            actual_value=actual, recorded_at="2026-02-17T00:00:00+00:00",
        )
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="5105571", stat="threes",
        predicted_value=3.0, created_at=PRE_TIP,
    )
    store.insert_player_outcome(
        db_path, game_id="g1", player_id="5105571", stat="threes",
        actual_value=4.0, recorded_at="2026-02-17T00:00:00+00:00",
    )


def _predict(db_path, player_id, stat, value=25.0):
    store.insert_player_prediction(
        db_path, game_id="g2", player_id=player_id, stat=stat,
        predicted_value=value, created_at=UPCOMING,
    )


def test_a_prop_row_carries_the_mae_for_its_own_stat(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    _seed_resolved(db_path)
    _predict(db_path, "203999", "points")

    rows = client.get("/games/g2/players").json()

    assert [(r["stat"], r["mae"]) for r in rows] == [("points", pytest.approx(3.0))]


def test_each_stat_carries_its_own_error_estimate(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    _seed_resolved(db_path)
    for stat in ("points", "threes", "rebounds"):
        _predict(db_path, "203999", stat)

    by_stat = {r["stat"]: r["mae"] for r in client.get("/games/g2/players").json()}

    assert by_stat == {
        "points": pytest.approx(3.0),   # two resolved rows, mean of 4 and 2
        "threes": pytest.approx(1.0),   # one resolved row
        "rebounds": None,               # predicted, never resolved
    }


def test_a_stat_with_no_resolved_rows_is_served_as_null_not_zero(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    _predict(db_path, "3934672", "rebounds")

    row = client.get("/games/g2/players").json()[0]

    assert row["mae"] is None
    assert row["mae"] != 0.0


def test_the_mae_does_not_change_when_an_unresolved_prediction_is_added(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    _seed_resolved(db_path)
    _predict(db_path, "203999", "points")
    before = client.get("/games/g2/players").json()[0]["mae"]

    # A wildly wrong prediction with no actual behind it must move nothing.
    _predict(db_path, "203999", "points", value=99.0)

    assert client.get("/games/g2/players").json()[0]["mae"] == before == pytest.approx(3.0)


def test_a_backtest_row_graded_after_the_fact_does_not_reach_the_mae(client, tmp_path):
    # The same look-forward leak the CFB record has: a prediction written after
    # the game was played, scored against that game's result. Judged nowhere.
    db_path = tmp_path / "tracking.db"
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="5105571", stat="points",
        predicted_value=50.0, created_at=POST_TIP,
    )
    store.insert_player_outcome(
        db_path, game_id="g1", player_id="5105571", stat="points",
        actual_value=10.0, recorded_at="2026-02-17T00:00:00+00:00",
    )
    store.insert_player_prediction(
        db_path, game_id="g2", player_id="203999", stat="points",
        predicted_value=25.0, created_at=UPCOMING,
    )

    assert client.get("/games/g2/players").json()[0]["mae"] is None


def test_only_the_latest_pre_tip_prediction_is_graded(client, tmp_path):
    # The MAE describes the number the endpoint actually serves, so it grades
    # the same pick the row shows, not every snapshot ever written.
    db_path = tmp_path / "tracking.db"
    for created_at, predicted in ((PRE_TIP, 1.0), ("2026-02-15T00:00:00+00:00", 2.0)):
        store.insert_player_prediction(
            db_path, game_id="g1", player_id="5105571", stat="points",
            predicted_value=predicted, created_at=created_at,
        )
    store.insert_player_outcome(
        db_path, game_id="g1", player_id="5105571", stat="points",
        actual_value=20.0, recorded_at="2026-02-17T00:00:00+00:00",
    )
    store.insert_player_prediction(
        db_path, game_id="g2", player_id="203999", stat="points",
        predicted_value=25.0, created_at=UPCOMING,
    )

    assert client.get("/games/g2/players").json()[0]["mae"] == pytest.approx(19.0)


def test_an_out_players_row_is_absent_from_the_ranking(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    for player_id in ("5105571", "3934672"):
        store.insert_player_prediction(
            db_path, game_id="g2", player_id=player_id, stat="points",
            predicted_value=25.0, created_at=UPCOMING,
        )
    _injuries({"team": "BOS", "player_id": "5105571", "player_name": "Jayson Tatum",
               "status": "Out", "dated": "2026-09-21T19:50Z"})

    rows = client.get("/games/g2/players").json()

    assert [r["player_id"] for r in rows] == ["3934672"]


def test_the_out_player_is_surfaced_once_with_a_source_and_a_date(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    for stat in ("points", "rebounds", "assists", "threes"):
        store.insert_player_prediction(
            db_path, game_id="g2", player_id="5105571", stat=stat,
            predicted_value=25.0, created_at=UPCOMING,
        )
    _injuries({"team": "BOS", "player_id": "5105571", "player_name": "Jayson Tatum",
               "status": "Out", "dated": "2026-09-21T19:50Z"})

    out = client.get("/games/g2/players/out").json()

    assert client.get("/games/g2/players").json() == []
    assert len(out) == 1  # four stat rows, one player
    assert out[0]["player_name"] == "Jayson Tatum"
    assert out[0]["team"] == "BOS"
    assert out[0]["status"] == "Out"
    assert "ESPN" in out[0]["source"]
    assert out[0]["dated"] == "2026-09-21T19:50Z"


def test_an_unresolvable_injury_removes_nobody(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    for player_id in ("5105571", "3934672"):
        store.insert_player_prediction(
            db_path, game_id="g2", player_id=player_id, stat="points",
            predicted_value=25.0, created_at=UPCOMING,
        )
    # Blank id: the report could not be resolved to a person, so it removes
    # nobody. A false removal is worse than a missed one.
    _injuries({"team": "BOS", "player_id": "", "player_name": "Jayson Tatum",
               "status": "Out", "dated": "2026-09-21T19:50Z"})

    rows = client.get("/games/g2/players").json()

    assert sorted(r["player_id"] for r in rows) == ["3934672", "5105571"]
    assert client.get("/games/g2/players/out").json() == []


def test_an_injury_for_someone_not_in_this_game_removes_nobody(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    store.insert_player_prediction(
        db_path, game_id="g2", player_id="5105571", stat="points",
        predicted_value=25.0, created_at=UPCOMING,
    )
    _injuries({"team": "BOS", "player_id": "9999999", "player_name": "LeBron James",
               "status": "Out", "dated": "2026-09-21T19:50Z"})

    assert len(client.get("/games/g2/players").json()) == 1
    assert client.get("/games/g2/players/out").json() == []


def test_a_day_to_day_player_stays_in_the_ranking(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    store.insert_player_prediction(
        db_path, game_id="g2", player_id="5105571", stat="points",
        predicted_value=25.0, created_at=UPCOMING,
    )
    _injuries({"team": "BOS", "player_id": "5105571", "player_name": "Jayson Tatum",
               "status": "Day-To-Day", "dated": "2026-09-21T19:50Z"})

    assert len(client.get("/games/g2/players").json()) == 1
    assert client.get("/games/g2/players/out").json() == []


def test_the_out_endpoint_404s_an_unknown_game(client):
    assert client.get("/games/nope/players/out").status_code == 404


def test_a_row_serves_no_probability_field(client, tmp_path):
    # There is no calibrated probability for this model to serve, so there is
    # no field for one. predict_double_double_probability exists in
    # player_props.py and has zero callers; wiring it up is not this phase.
    db_path = tmp_path / "tracking.db"
    store.insert_player_prediction(
        db_path, game_id="g2", player_id="3934672", stat="points",
        predicted_value=25.0, created_at=UPCOMING,
    )

    row = client.get("/games/g2/players").json()[0]

    assert set(row) == {
        "player_id", "player_name", "stat", "predicted_value",
        "actual_value", "rebuilt", "mae",
    }
    assert not any(
        token in field.lower() for field in row for token in ("prob", "pct", "percent", "chance", "odds")
    )


def test_the_out_entry_serves_no_probability_field(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    store.insert_player_prediction(
        db_path, game_id="g2", player_id="5105571", stat="points",
        predicted_value=25.0, created_at=UPCOMING,
    )
    _injuries({"team": "BOS", "player_id": "5105571", "player_name": "Jayson Tatum",
               "status": "Out", "dated": "2026-09-21T19:50Z"})

    entry = client.get("/games/g2/players/out").json()[0]

    assert set(entry) == {"player_id", "player_name", "team", "status", "source", "dated"}


def test_the_props_endpoint_refuses_to_serve_ranked_rows_when_the_feed_fails(tmp_path, monkeypatch):
    # Decision 6: rows do not ship without the gate. A feed we cannot read is
    # not a gate that came back empty, and the two must not look alike.
    from nba_predictor.api import deps as deps_module

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(json.dumps(SCHEDULE))
    _predict(db_path, "203999", "points")

    def _boom():
        raise RuntimeError("injury feed unreachable")

    # Overriding the dependency is not how the real feed fails -- the real
    # failure is inside deps.get_injury_report, which is what must convert it
    # into a 503 rather than a 500 or a silently ungated ranking.
    monkeypatch.setattr("nba_predictor.data.espn.get_injuries", _boom)
    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path
    # Drop any override from the fixture so the real dependency runs.
    app.dependency_overrides.pop(deps.get_injury_report, None)
    try:
        response = TestClient(app, raise_server_exceptions=False).get("/games/g2/players")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "availability" in response.json()["detail"].lower()


def test_an_empty_injury_report_is_served_normally_not_refused(tmp_path, monkeypatch):
    # An empty report and an unreadable feed are different facts: the first
    # means nobody is out, the second means nobody checked.
    monkeypatch.setattr("nba_predictor.data.espn.get_injuries", lambda: [])
    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(json.dumps(SCHEDULE))
    _predict(db_path, "203999", "points")

    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path
    app.dependency_overrides.pop(deps.get_injury_report, None)
    try:
        response = TestClient(app, raise_server_exceptions=False).get("/games/g2/players")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert len(response.json()) == 1
