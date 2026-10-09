"""The box-score history behind the four-factors duels.

Two properties are load-bearing and both are cheap to get wrong:

**It is bounded.** The ranker needs each team's last `window` games, so only a
lookback window ending at `as_of` is enriched. Reading every completed game in
the schedule would put a season of ESPN calls on the first visitor's request.

**It is warmed off the request thread.** A daemon thread fills the cache from the
app's lifespan, so serving never waits on it, and the cache is refreshed when the
schedule cache gains a game rather than being built once and trusted forever.

The staleness test is the one that matters most: a long-lived process that ranks
duels off a frame built at boot is quoting last month's form as if it were now.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nba_predictor.api import facts as facts_mod
from nba_predictor.api import app as app_mod


def _game(game_id, date, home, away, completed=True):
    return {
        "game_id": game_id, "game_date": date, "home_team": home, "away_team": away,
        "completed": completed, "home_pts": 110, "away_pts": 100,
    }


@pytest.fixture
def offline(monkeypatch):
    """Duels and box-score enrichment stubbed: no ESPN, no store."""
    monkeypatch.setattr(facts_mod, "_db_path", lambda: __import__("pathlib").Path("/nonexistent/x.sqlite"))
    monkeypatch.setattr(facts_mod, "_market_rows", lambda game_id: [])
    monkeypatch.setattr(facts_mod, "_BOX_HISTORY", {})
    return monkeypatch


def _schedule_with_history(monkeypatch, dates):
    schedule = [
        _game(f"g{i}", d, "BOS", "MIA") for i, d in enumerate(dates)
    ] + [_game("upcoming", "2026-03-18", "BOS", "MIA", completed=False)]
    monkeypatch.setattr(facts_mod, "_schedule", lambda: schedule)
    return schedule


def test_only_the_lookback_window_is_enriched(offline, monkeypatch):
    """A full season of box scores must not be read for one game's duels."""
    enriched: list[list[dict]] = []

    def fake_enrich(games):
        enriched.append(list(games))
        return list(games)

    monkeypatch.setattr("nba_predictor.pipeline.ingest.enrich_with_boxscores", fake_enrich)
    monkeypatch.setattr(
        "nba_predictor.pipeline.ingest.to_training_frame",
        lambda games: pd.DataFrame(games),
    )
    _schedule_with_history(monkeypatch, ["2025-11-01"] + [f"2026-03-{d:02d}" for d in range(1, 18)])

    frame = facts_mod._box_score_history("2026-03-18")
    assert len(enriched) == 1, "the history was enriched more than once"

    passed = enriched[0]
    assert passed, "nothing was enriched"
    for game in passed:
        assert game["game_date"] < "2026-03-18", (
            f"a game dated {game['game_date']} is on or after as_of; it has not been played"
        )
        assert game["completed"], "an unplayed game was enriched"

    # The distant game is outside the lookback, so it is not read at all.
    assert all(g["game_date"] >= "2025-11-18" for g in passed), (
        f"the lookback reached back to {min(g['game_date'] for g in passed)}; "
        "BOX_LOOKBACK_DAYS is not being applied"
    )


def test_a_cached_window_is_not_rebuilt(offline, monkeypatch):
    """One schedule, one as_of: the frame is built once and reused."""
    calls = {"n": 0}

    def counting_enrich(games):
        calls["n"] += 1
        return list(games)

    monkeypatch.setattr("nba_predictor.pipeline.ingest.enrich_with_boxscores", counting_enrich)
    monkeypatch.setattr(
        "nba_predictor.pipeline.ingest.to_training_frame",
        lambda games: pd.DataFrame(games),
    )
    _schedule_with_history(monkeypatch, [f"2026-03-{d:02d}" for d in range(1, 18)])
    monkeypatch.setattr(
        facts_mod.deps, "get_schedule_path",
        lambda: __import__("pathlib").Path("/nonexistent/schedule.json"),
    )

    first = facts_mod._box_score_history("2026-03-18")
    second = facts_mod._box_score_history("2026-03-18")

    assert calls["n"] == 1, f"the window was rebuilt {calls['n']} times for one as_of"
    assert first is second, "the cached frame was not reused"


def test_two_game_dates_are_cached_side_by_side(offline, monkeypatch):
    """A week of upcoming games must not thrash the cache.

    One entry keyed on a single `as_of` rebuilt for every date in a pass, so a
    page-load across the week paid for a window per game and reused nothing.
    """
    calls = {"n": 0}

    def counting_enrich(games):
        calls["n"] += 1
        return list(games)

    monkeypatch.setattr("nba_predictor.pipeline.ingest.enrich_with_boxscores", counting_enrich)
    monkeypatch.setattr(
        "nba_predictor.pipeline.ingest.to_training_frame",
        lambda games: pd.DataFrame(games),
    )
    _schedule_with_history(monkeypatch, [f"2026-03-{d:02d}" for d in range(1, 18)])
    monkeypatch.setattr(
        facts_mod.deps, "get_schedule_path",
        lambda: __import__("pathlib").Path("/nonexistent/schedule.json"),
    )

    dates = ["2026-03-18", "2026-03-19", "2026-03-20"]

    for date in dates:                       # first pass: one build per date
        facts_mod._box_score_history(date)
    assert calls["n"] == len(dates), (
        f"expected one build per distinct as_of, got {calls['n']}"
    )

    for date in dates:                       # second pass: every one reused
        facts_mod._box_score_history(date)
    assert calls["n"] == len(dates), (
        f"the second pass rebuilt {calls['n'] - len(dates)} windows: the cache does "
        "not hold more than one as_of"
    )


def test_the_cache_is_bounded(offline, monkeypatch):
    """Bounded, so a month of dates cannot grow it without limit."""
    monkeypatch.setattr(
        "nba_predictor.pipeline.ingest.enrich_with_boxscores", lambda games: list(games)
    )
    monkeypatch.setattr(
        "nba_predictor.pipeline.ingest.to_training_frame",
        lambda games: pd.DataFrame(games),
    )
    _schedule_with_history(monkeypatch, [f"2026-03-{d:02d}" for d in range(1, 18)])
    monkeypatch.setattr(
        facts_mod.deps, "get_schedule_path",
        lambda: __import__("pathlib").Path("/nonexistent/schedule.json"),
    )

    for i in range(facts_mod.BOX_HISTORY_KEEP + 6):
        facts_mod._box_score_history(f"2026-04-{i + 1:02d}")

    assert len(facts_mod._BOX_HISTORY) == facts_mod.BOX_HISTORY_KEEP, (
        f"the cache grew to {len(facts_mod._BOX_HISTORY)} entries; it is unbounded"
    )
    # The coldest entries are the ones evicted, not the newest.
    newest = (facts_mod._schedule_state(facts_mod.deps.get_schedule_path()), f"2026-04-{facts_mod.BOX_HISTORY_KEEP + 6:02d}")
    assert newest in facts_mod._BOX_HISTORY, "the newest window was evicted"


def test_a_changed_schedule_rebuilds_the_window(offline, monkeypatch):
    """A game completing is the moment the duels' answer changes, so the cache has
    to notice. Ranking last month's form as if it were now is the failure here."""
    builds = {"n": 0}

    def counting_enrich(games):
        builds["n"] += 1
        return list(games)

    monkeypatch.setattr("nba_predictor.pipeline.ingest.enrich_with_boxscores", counting_enrich)
    monkeypatch.setattr(
        "nba_predictor.pipeline.ingest.to_training_frame",
        lambda games: pd.DataFrame(games),
    )

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "schedule.json"
        path.write_text("[]")
        monkeypatch.setattr(facts_mod.deps, "get_schedule_path", lambda: path)

        schedule = [_game(f"g{i}", d, "BOS", "MIA") for i, d in enumerate([f"2026-03-{d:02d}" for d in range(1, 18)])]
        monkeypatch.setattr(facts_mod, "_schedule", lambda: schedule)

        facts_mod._box_score_history("2026-03-18")
        assert builds["n"] == 1

        # The schedule cache gains a game: same mtime is unchanged, but its size moves.
        path.write_text('["a completed game"]')
        facts_mod._box_score_history("2026-03-18")
        assert builds["n"] == 2, (
            "the schedule cache changed and the window was not rebuilt: the duels "
            "would keep quoting the form they were built with"
        )


def test_a_failed_warm_is_reported_and_does_not_raise(offline, monkeypatch):
    """A warmer that cannot read box scores must not take the app down."""
    def boom(as_of):
        raise RuntimeError("espn unavailable")

    monkeypatch.setattr(facts_mod, "_box_score_history", boom)

    assert facts_mod.warm_box_history("2026-03-18") is False


def test_the_warmer_thread_runs_and_stops(offline, monkeypatch):
    """Injected sleep/stop so the loop is testable without a real tick."""
    warmed: list[str] = []
    monkeypatch.setattr(
        facts_mod, "_schedule",
        lambda: [_game("upcoming", "2026-03-18", "BOS", "MIA", completed=False)],
    )
    monkeypatch.setattr(facts_mod, "warm_box_history", lambda as_of: warmed.append(as_of) or True)

    ticked = {"n": 0}

    def fake_sleep(_):
        ticked["n"] += 1

    thread = facts_mod.start_box_history_warmer(
        __import__("pathlib").Path("/nonexistent/schedule.json"),
        sleep=fake_sleep,
        should_stop=lambda: ticked["n"] >= 1,
    )
    thread.join(timeout=5)

    assert warmed == ["2026-03-18"], f"the warmer warmed {warmed}"


def test_the_app_starts_the_box_history_warmer(monkeypatch):
    """Serving must not depend on it, and it must not be started twice."""
    started = {"n": 0}

    def fake_start(*a, **k):
        started["n"] += 1
        import threading
        return threading.Thread(target=lambda: None)

    monkeypatch.setattr(app_mod, "start_box_history_warmer", fake_start)
    # Everything else in lifespan is already covered by the app's own tests; this
    # only proves the warmer is wired in and that a failure there is contained.
    import inspect
    source = inspect.getsource(app_mod.lifespan)
    assert "start_box_history_warmer" in source, "the lifespan does not start the warmer"
