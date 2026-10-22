"""GET /picks/history: the feed the paper ledger's CI path reads.

The ledger's CI run has no tracking DB, so it fetches the pick history from
this route instead. That makes the route part of the ledger's live path, and
a route that raises on every call takes the whole CI feed with it.

This file pins three things:

  * the route returns rows (and does not raise);
  * the rows carry the per-game probabilities the ledger pairs with a price,
    including the margin's sigma, without which a spread rung cannot be
    priced at its own line;
  * `since` filters, so a caller can ask for what is new.
"""

import json

from fastapi.testclient import TestClient

from nba_predictor import config
from nba_predictor.api.app import app
from nba_predictor.tracking import store


def _point_at(tmp_path, monkeypatch):
    """Point the route's inputs at a tmp dir with one prediction in it."""
    # Three separate roots: the tracking DB, the schedule cache (under
    # DATA_DIR) and the manifest (under PROJECT_ROOT).
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    db_path = tmp_path / "tracking.db"
    monkeypatch.setattr(config, "TRACKING_DB_PATH", db_path)

    schedule_path = tmp_path / "cache" / "schedule" / "games.json"
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_path.write_text(json.dumps([
        {"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"},
    ]))

    manifest_path = tmp_path / "models" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "model_version": "v1", "trained_at": "2026-10-18T00:00:00+00:00", "models": [],
        "metrics": {"total": {"residual_sigma": 15.0}, "margin": {"residual_sigma": 12.0}},
    }))

    store.init_db(db_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-10-18T12:00:00+00:00", model_version="v1",
        home_win_prob=0.60, predicted_margin=3.0, predicted_total=220.0,
    )
    return db_path


def test_the_route_answers_with_the_picks_it_has(tmp_path, monkeypatch):
    """The route must not raise.

    It used to reference the manifest's sigma helpers without importing them,
    so every call raised NameError -- and the ledger's CI run fetches its feed
    from here, so the failure was invisible until the workflow ran.
    """
    _point_at(tmp_path, monkeypatch)
    client = TestClient(app)
    response = client.get("/picks/history")
    assert response.status_code == 200
    body = response.json()
    assert "rows" in body


def test_the_rows_carry_what_the_ledger_pairs_with_a_price(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    client = TestClient(app)
    rows = client.get("/picks/history").json()["rows"]
    assert len(rows) == 1
    row = rows[0]
    assert row["game_id"] == "g1"
    assert row["home_win_prob"] == 0.60
    assert row["predicted_margin"] == 3.0
    assert row["predicted_total"] == 220.0
    assert row["total_sigma"] == 15.0
    # The margin's sigma, so a spread rung can be priced at its own line.
    assert row["margin_sigma"] == 12.0
    assert row["game_date"] == "2026-11-01"
    assert row["home_team"] == "BOS" and row["away_team"] == "MIA"


def test_since_filters_to_what_is_new(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    client = TestClient(app)
    assert len(client.get("/picks/history").json()["rows"]) == 1
    assert client.get("/picks/history?since=2026-10-19T00:00:00%2B00:00").json()["rows"] == []
    assert len(client.get("/picks/history?since=2026-10-01T00:00:00%2B00:00").json()["rows"]) == 1


def test_a_manifest_with_no_margin_sigma_leaves_the_field_out(tmp_path, monkeypatch):
    """No margin sigma is absent, not zero.

    Zero would be a claim that the margin is certain, which is a different
    claim from "we do not know", and the ledger would price a spread rung
    from it.
    """
    _point_at(tmp_path, monkeypatch)
    manifest_path = tmp_path / "models" / "manifest.json"
    manifest_path.write_text(json.dumps({
        "model_version": "v1", "trained_at": "2026-10-18T00:00:00+00:00", "models": [],
        "metrics": {"total": {"residual_sigma": 15.0}},
    }))
    client = TestClient(app)
    row = client.get("/picks/history").json()["rows"][0]
    assert "margin_sigma" not in row or row["margin_sigma"] is None
