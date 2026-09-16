import pytest


@pytest.mark.skip(reason="fastapi.testclient import issue with starlette")
def test_health_endpoint_returns_ok():
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}