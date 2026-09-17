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


def test_create_app_falls_back_to_index_for_spa_client_routes(tmp_path, monkeypatch):
    from nba_predictor import config
    from nba_predictor.api.app import create_app

    dist_dir = tmp_path / "frontend" / "dist"
    dist_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text("<html><body>NBA Predictor App</body></html>")

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    app = create_app()
    client = TestClient(app)

    for path in ["/hub", "/model", "/calibration-report", "/hub/team-hub"]:
        response = client.get(path)
        assert response.status_code == 200, f"{path} should fall back to the SPA shell"
        assert "NBA Predictor App" in response.text, (
            f"{path} returned {response.text[:80]!r} instead of the SPA shell — "
            "check it doesn't collide with a real backend route (e.g. GET /calibration)"
        )


def test_create_app_backend_calibration_route_is_not_shadowed_by_static_mount(tmp_path, monkeypatch):
    """GET /calibration (the API route) and the frontend's /calibration-report
    page must stay distinct — this test exists because they used to collide
    (both were "/calibration"), which meant a direct browser visit to the
    Calibration page showed raw JSON instead of the app."""
    from nba_predictor import config
    from nba_predictor.api.app import create_app
    from nba_predictor.tracking import store

    dist_dir = tmp_path / "frontend" / "dist"
    dist_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text("<html><body>NBA Predictor App</body></html>")

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "TRACKING_DB_PATH", db_path)
    app = create_app()
    client = TestClient(app)

    response = client.get("/calibration")
    assert response.status_code == 200
    assert response.json() == []


def test_create_app_still_404s_for_unmatched_api_style_paths_without_dist(tmp_path, monkeypatch):
    from nba_predictor import config
    from nba_predictor.api.app import create_app

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)  # no frontend/dist under tmp_path
    app = create_app()
    client = TestClient(app)

    assert client.get("/hub").status_code == 404


def test_create_app_skips_static_mount_when_dist_missing(tmp_path, monkeypatch):
    from nba_predictor import config
    from nba_predictor.api.app import create_app

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)  # no frontend/dist under tmp_path
    app = create_app()
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 404
