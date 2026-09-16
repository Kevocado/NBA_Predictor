from fastapi.testclient import TestClient


def test_create_app_serves_frontend_index_when_dist_exists(tmp_path, monkeypatch):
    from nba_predictor import config
    from nba_predictor.api.app import create_app

    dist_dir = tmp_path / "frontend" / "dist"
    dist_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text("<html><body>NBA Predictor App</body></html>")

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    app = create_app()
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "NBA Predictor App" in response.text


def test_create_app_health_route_takes_precedence_over_static_mount(tmp_path, monkeypatch):
    from nba_predictor import config
    from nba_predictor.api.app import create_app

    dist_dir = tmp_path / "frontend" / "dist"
    dist_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text("<html></html>")

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    app = create_app()
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_app_skips_static_mount_when_dist_missing(tmp_path, monkeypatch):
    from nba_predictor import config
    from nba_predictor.api.app import create_app

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)  # no frontend/dist under tmp_path
    app = create_app()
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 404
