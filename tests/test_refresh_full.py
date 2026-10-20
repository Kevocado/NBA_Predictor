"""The daily refresh entry point: ingest, then the odds-derived markets; a markets fault never fails the run."""
from nba_predictor.pipeline import refresh_full


def test_it_runs_the_ingest_then_the_market_pass(monkeypatch):
    calls = []
    monkeypatch.setattr(refresh_full, "run_ingest", lambda start, end, **kw: calls.append(("ingest", start, end)) or {"games_found": 3})
    monkeypatch.setattr(refresh_full, "refresh_market_predictions", lambda *a, **k: calls.append(("markets",)) or 7)
    monkeypatch.setattr(refresh_full, "load_schedule", lambda path: [])
    summary = refresh_full.run_full_refresh(log=lambda _m: None)
    assert [c[0] for c in calls] == ["ingest", "markets"]
    assert calls[0][1] == refresh_full.default_window_start() and calls[0][2] == refresh_full.default_window_end()
    assert summary["market_predictions_stored"] == 7


def test_a_failing_market_pass_does_not_fail_the_refresh(monkeypatch):
    monkeypatch.setattr(refresh_full, "run_ingest", lambda *a, **k: {"games_found": 3})
    monkeypatch.setattr(refresh_full, "load_schedule", lambda path: [])

    def boom(*a, **k):
        raise RuntimeError("odds api down")

    monkeypatch.setattr(refresh_full, "refresh_market_predictions", boom)
    summary = refresh_full.run_full_refresh(log=lambda _m: None)
    assert summary["games_found"] == 3 and "odds api down" in summary["market_error"]
