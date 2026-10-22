"""The CI snapshot job has no tracking DB, so it fetches the pick history from the live API."""
import gzip
import io
import json

from nba_predictor import public_snapshot as ps


def _api(rows):
    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return lambda req, timeout=None: Resp(json.dumps({"rows": rows}).encode())


def _row(game, created, **extra):
    return {"game_id": game, "created_at": created, "game_date": "2026-10-23", "home_team": "BOS", "away_team": "MIA",
            "home_win_prob": 0.6, "predicted_margin": 3.0, "predicted_total": 224.0, "total_sigma": 18.7, "margin_sigma": 13.5,
            "model_version": "v1", **extra}


def _lines(path):
    return [json.loads(line) for line in gzip.open(path, "rt")]


def test_rows_are_appended_with_both_sigmas_and_nothing_is_rewritten(tmp_path, monkeypatch):
    history = tmp_path / "picks_history.jsonl.gz"
    monkeypatch.setattr("urllib.request.urlopen", _api([_row("g1", "2026-10-22T10:00:00+00:00")]))
    assert ps.fetch_and_append_picks("https://api.test", history) == 1
    monkeypatch.setattr("urllib.request.urlopen", _api([_row("g1", "2026-10-22T10:00:00+00:00"), _row("g1", "2026-10-22T14:00:00+00:00", predicted_total=226.0)]))
    assert ps.fetch_and_append_picks("https://api.test", history) == 1          # only the new version: the old row is untouched
    rows = _lines(history)
    assert [r["created_at"] for r in rows] == ["2026-10-22T10:00:00+00:00", "2026-10-22T14:00:00+00:00"]
    assert rows[0]["predicted_total"] == 224.0 and rows[0]["margin_sigma"] == 13.5 and rows[0]["total_sigma"] == 18.7


def test_the_request_asks_since_the_newest_pick_already_held_not_since_now(tmp_path, monkeypatch):
    history = tmp_path / "picks_history.jsonl.gz"
    seen = []
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: (seen.append(req.full_url), _api([])(req))[1])
    ps.fetch_and_append_picks("https://api.test/", history)                      # empty history: a two-week lookback, not everything
    assert "/picks/history?since=" in seen[-1]
    from urllib.parse import unquote
    from datetime import datetime, timedelta, timezone
    asked = datetime.fromisoformat(unquote(seen[-1].split("since=")[1]))
    assert timedelta(days=13) < datetime.now(timezone.utc) - asked < timedelta(days=15)
    monkeypatch.setattr("urllib.request.urlopen", _api([_row("g1", "2026-10-22T10:00:00+00:00")]))
    ps.fetch_and_append_picks("https://api.test", history)
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: (seen.append(req.full_url), _api([])(req))[1])
    ps.fetch_and_append_picks("https://api.test", history)
    assert seen[-1].endswith("/picks/history?since=2026-10-22T10%3A00%3A00%2B00%3A00")
