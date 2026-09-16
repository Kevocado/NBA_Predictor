def test_list_teams_returns_thirty_teams():
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import app

    client = TestClient(app)
    response = client.get("/teams")

    assert response.status_code == 200
    assert len(response.json()) == 30


def test_get_team_returns_matching_team():
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import app

    client = TestClient(app)
    response = client.get("/teams/BOS")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Boston Celtics"
    assert body["conference"] == "East"


def test_get_team_404_for_unknown_abbreviation():
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import app

    client = TestClient(app)
    response = client.get("/teams/ZZZ")

    assert response.status_code == 404
