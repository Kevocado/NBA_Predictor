"""The retrain must not be frozen.

Gap G3: models/manifest_history.jsonl held four entries with byte-identical
metrics, trained_at frozen at 2026-09-18 while the daily workflow claimed to
run. Diagnosis in docs/nba-retrain-diagnosis-2026-10.md: the workflow's rolling
60-day window could not cover a season, so to_training_frame() was left with one
row and chronological_split raised before any manifest was written.

These tests pin the two halves of that:

* the workflow window reaches back past the start of the previous NBA season,
  so it survives the offseason (where the bug bit), and
* two retrains on different training data append two *different* history
  entries -- the never-frozen regression test the plan asks for.
"""

import json
from pathlib import Path


def _games(n: int, *, seed: int, start: str = "2025-10-22") -> list[dict]:
    """n completed games with a full box score, seeded so two calls differ."""
    import pandas as pd

    teams = ["BOS", "MIA", "LAL", "GSW"]
    dates = pd.date_range(start, periods=n).astype(str)
    rows = []
    for i, game_date in enumerate(dates):
        home, away = teams[i % 4], teams[(i + 1) % 4]
        # seed shifts the scores, so the fitted models -- and therefore the
        # manifest metrics -- genuinely differ between calls.
        home_pts = 100 + ((i * 7 + seed * 13) % 30)
        away_pts = 100 + ((i * 5 + seed * 29) % 30)
        rows.append(
            {
                "game_id": f"g{seed}-{i}",
                "game_date": str(game_date),
                "home_team": home,
                "away_team": away,
                "completed": True,
                "home_pts": home_pts,
                "away_pts": away_pts,
                "home_fgm": 40, "home_fga": 88, "home_fg3m": 12,
                "home_tov": 11, "home_oreb": 9, "home_dreb": 32, "home_fta": 20,
                "away_fgm": 38, "away_fga": 90, "away_fg3m": 10,
                "away_tov": 13, "away_oreb": 10, "away_dreb": 30, "away_fta": 18,
                "home_win": int(home_pts > away_pts),
            }
        )
    return rows


def _workflow() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / ".github" / "workflows" / "refresh-data.yml"
    ).read_text()


def test_default_window_start_reaches_back_a_whole_season():
    """The bug: `date -d '60 days ago'`. During the offseason that window holds
    almost no completed games, to_training_frame() returns one row, and the run
    dies before writing a manifest. A training window has to reach back to the
    start of the previous season at minimum -- on every day of the year, not
    just while the season is running."""
    from datetime import date

    from nba_predictor.pipeline.ingest import default_window_start

    # Offseason is when the 60-day window broke: the regular season had not
    # started, so there was nothing to train on.
    for today in (date(2026, 10, 5), date(2027, 1, 20), date(2027, 6, 15)):
        start = date.fromisoformat(default_window_start(today))
        assert start <= date(today.year - 1, 10, 1), (
            f"window starting {start} on {today} cannot cover a season -- the "
            "offseason run trains on nothing (G3)"
        )


def test_workflow_does_not_pass_a_rolling_window():
    """The window has to come from the season-anchored default. A hardcoded
    start date in the workflow would freeze again next October."""
    # Only the ingest invocation is checked, not the file's prose: a comment
    # explaining the old bug must not be able to fail this test.
    invoke = "\n".join(
        line for line in _workflow().splitlines() if "nba_predictor.pipeline.ingest" in line
    )
    assert "60 days ago" not in invoke, "the training window is still rolling (G3)"
    assert "--start" not in invoke, (
        "the workflow still overrides --start; it must use the season-anchored "
        "default or the window freezes again next season"
    )


def test_two_retrains_on_different_data_produce_different_metrics(tmp_path):
    """The never-frozen regression test.

    Byte-identical metrics across retrains is the signature of a retrain that
    ingests nothing new. Two different training sets must move them.
    """
    import pandas as pd

    from nba_predictor.pipeline.retrain import run_retrain_pipeline

    models_dir = tmp_path / "models"
    manifest_path = models_dir / "manifest.json"
    history_path = models_dir / "manifest_history.jsonl"

    def run(training_games, trained_at):
        return run_retrain_pipeline(
            pd.DataFrame(training_games),
            models_dir,
            model_version=f"v{trained_at}",
            trained_at=trained_at,
        )

    m1 = run(_games(120, seed=1), "2026-10-05T00:00:00+00:00")
    m2 = run(_games(120, seed=2), "2026-10-06T00:00:00+00:00")

    assert m1["metrics"] != m2["metrics"], (
        "retrain is frozen: metrics byte-identical across different training data"
    )

    entries = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    assert len(entries) == 2, f"expected two history entries, got {len(entries)}"
    assert entries[0]["metrics"] != entries[1]["metrics"], (
        "manifest_history.jsonl entries are byte-identical across different data"
    )
    assert entries[0]["trained_at"] != entries[1]["trained_at"]

    # And what actually ships is the second manifest, not the first.
    assert json.loads(manifest_path.read_text())["trained_at"] == m2["trained_at"]


def test_ingest_writes_the_training_cache_the_admin_endpoint_reads(tmp_path, monkeypatch):
    """api/deps.py reads data/cache/training/games.json; nothing in src/ ever
    wrote it, so POST /retrain 400s everywhere -- including the VPS. The file
    has to exist with the box-score fields the retrain needs."""
    from nba_predictor import config
    from nba_predictor.pipeline import ingest

    games = _games(40, seed=3)
    # Stubs sit on run_ingest's own seams rather than on ESPN's surface: a
    # hand-rolled ESPN stub that disagrees with the real API would make this
    # test agree with the bug (spec section 12.1).
    monkeypatch.setattr(ingest, "fetch_schedule_range", lambda start, end: games)
    monkeypatch.setattr(ingest, "enrich_with_boxscores", lambda gs: gs)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)

    # Skip the model half of the pipeline -- this test is about the cache file
    # run_ingest leaves behind for POST /retrain, and training is covered above.
    monkeypatch.setattr(
        ingest, "run_retrain_pipeline",
        lambda games, models_dir, model_version, trained_at: {"metrics": {}},
    )
    monkeypatch.setattr(
        ingest, "train_player_prop_models",
        lambda *a, **k: {"metrics": {}},
    )

    # skip_predictions is what the workflow passes; this test is about the cache
    # file, not the scoring tail (which needs player box scores it has no use for).
    result = ingest.run_ingest(
        "2025-10-22",
        "2025-12-01",
        player_hub_days=0,
        skip_predictions=True,
        db_path=tmp_path / "t.db",
    )

    assert result["training_games_written"] == len(games)

    training_path = tmp_path / "cache" / "training" / "games.json"
    assert training_path.exists(), "nothing writes the cache POST /retrain reads"
    rows = json.loads(training_path.read_text())
    assert len(rows) == len(games)
    # The box fields are the whole point -- to_schedule_cache drops them.
    assert "home_fgm" in rows[0], "training cache has no box-score fields"
    assert rows[0]["home_fgm"] == 40