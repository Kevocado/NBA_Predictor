import numpy as np
import pandas as pd


def _synthetic_games(n=60, seed=2):
    rng = np.random.default_rng(seed)
    teams = ["BOS", "MIA", "LAL", "GSW"]
    dates = pd.date_range("2026-10-21", periods=n).astype(str)
    rows = []
    for i, game_date in enumerate(dates):
        home, away = teams[i % 4], teams[(i + 1) % 4]
        rows.append(
            {
                "game_id": f"g{i}", "game_date": game_date, "home_team": home, "away_team": away,
                "home_pts": 110, "away_pts": 105,
                "home_fgm": 40, "home_fga": 88, "home_fg3m": 12, "home_tov": 11, "home_oreb": 9, "home_dreb": 32, "home_fta": 20,
                "away_fgm": 38, "away_fga": 90, "away_fg3m": 10, "away_tov": 13, "away_oreb": 10, "away_dreb": 30, "away_fta": 18,
                "home_win": int(rng.random() > 0.4),
            }
        )
    return pd.DataFrame(rows)


def test_run_retrain_pipeline_produces_models_and_manifest(tmp_path):
    from nba_predictor.pipeline.retrain import run_retrain_pipeline

    games = _synthetic_games()
    models_dir = tmp_path / "models"

    manifest = run_retrain_pipeline(games, models_dir, model_version="v-test", trained_at="2026-11-01T00:00:00")

    assert manifest["model_version"] == "v-test"
    assert set(manifest["models"]) == {"win_probability", "margin", "total"}
    assert (models_dir / "win_probability_model.pkl").exists()
    assert (models_dir / "margin_model.pkl").exists()
    assert (models_dir / "total_model.pkl").exists()
    assert (models_dir / "manifest.json").exists()
    assert (models_dir / "manifest_history.jsonl").exists()
    assert "accuracy" in manifest["metrics"]["win_probability"]


def test_run_retrain_pipeline_appends_history_across_calls(tmp_path):
    from nba_predictor.pipeline.retrain import run_retrain_pipeline

    games = _synthetic_games()
    models_dir = tmp_path / "models"

    run_retrain_pipeline(games, models_dir, model_version="v1", trained_at="2026-11-01T00:00:00")
    run_retrain_pipeline(games, models_dir, model_version="v2", trained_at="2026-11-02T00:00:00")

    lines = (models_dir / "manifest_history.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
