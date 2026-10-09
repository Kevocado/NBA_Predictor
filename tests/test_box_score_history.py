"""The box-score history behind the four-factors duels.

Three properties are load-bearing and all are cheap to get wrong:

**It is bounded.** The ranker needs each team's last `window` games, so only a
lookback window ending at `as_of` is enriched. Reading every completed game in
the schedule would put a season of ESPN calls on one request.

**It never builds on a request thread.** `_box_score_history` reads the cache and
returns an empty frame when cold -- "no data, no signal". `warm_box_history` is
what builds, from a daemon thread, for every upcoming date.

**It is refreshed.** A game completing is the moment the duels' answer changes,
so the cache is invalidated on the schedule cache's (mtime, size). A long-lived
process that ranked duels off a frame built at boot would be quoting last month's
form as if it were now.
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
    """Nothing real is read: the schedule is doubled and the cache starts empty."""
    monkeypatch.setattr(facts_mod, "_db_path", lambda: __import__("pathlib").Path("/nonexistent/x.sqlite"))
    monkeypatch.setattr(facts_mod, "_market_rows", lambda game_id: [])
    monkeypatch.setattr(facts_mod, "_BOX_HISTORY", {})
    monkeypatch.setattr(
        facts_mod.deps, "get_schedule_path",
        lambda: __import__("pathlib").Path("/nonexistent/schedule.json"),
    )
    return monkeypatch


@pytest.fixture
def counting_enrich(monkeypatch):
    """`enrich_with_boxscores` and `to_training_frame` stubbed, and the enrich
    calls counted. No ESPN, no network."""
    calls = {"n": 0, "passed": []}

    def enrich(games):
        calls["n"] += 1
        calls["passed"].append(list(games))
        return list(games)

    monkeypatch.setattr("nba_predictor.pipeline.ingest.enrich_with_boxscores", enrich)
    monkeypatch.setattr(
        "nba_predictor.pipeline.ingest.to_training_frame",
        lambda games: pd.DataFrame(games),
    )
    return calls


def _schedule_with_history(monkeypatch, dates):
    schedule = [_game(f"g{i}", d, "BOS", "MIA") for i, d in enumerate(dates)]
    schedule.append(_game("upcoming", "2026-03-18", "BOS", "MIA", completed=False))
    monkeypatch.setattr(facts_mod, "_schedule", lambda: schedule)
    return schedule


DATES = [f"2026-03-{d:02d}" for d in range(1, 18)]


# --- the bound -----------------------------------------------------------

def test_only_the_lookback_window_is_enriched(offline, counting_enrich):
    """A full season of box scores must not be read for one game's duels."""
    _schedule_with_history(offline, ["2025-11-01"] + DATES)

    facts_mod.warm_box_history("2026-03-18")

    assert counting_enrich["n"] == 1, f"the window was enriched {counting_enrich['n']} times"
    passed = counting_enrich["passed"][0]
    assert passed, "nothing was enriched"
    for game in passed:
        assert game["game_date"] < "2026-03-18", (
            f"a game dated {game['game_date']} is on or after as_of; it has not been played"
        )
        assert game["completed"], "an unplayed game was enriched"
    assert all(g["game_date"] >= "2025-11-18" for g in passed), (
        f"the lookback reached back to {min(g['game_date'] for g in passed)}; "
        "BOX_LOOKBACK_DAYS is not being applied"
    )


# --- never build on a request thread -------------------------------------

def test_a_cold_request_yields_no_history_and_builds_nothing(offline, counting_enrich):
    """The request thread must not do the I/O, even when the cache is empty."""
    _schedule_with_history(offline, DATES)

    frame = facts_mod._box_score_history("2026-03-18")

    assert frame is not None and len(frame) == 0, (
        "a cold window returned a frame: the request thread built it"
    )
    assert counting_enrich["n"] == 0, (
        f"the request thread enriched {counting_enrich['n']} times; that is the I/O "
        "CodeRabbit flagged"
    )


def test_a_warm_request_returns_the_frame_it_was_given(offline, counting_enrich):
    _schedule_with_history(offline, DATES)

    facts_mod.warm_box_history("2026-03-18")

    frame = facts_mod._box_score_history("2026-03-18")
    assert len(frame) == len(DATES), (
        f"the request returned {len(frame)} rows for a warmed {len(DATES)}-game window"
    )


# --- the cache ------------------------------------------------------------

def test_a_warmed_window_is_not_rebuilt(offline, counting_enrich):
    """One schedule, one as_of: the frame is built once and reused."""
    _schedule_with_history(offline, DATES)

    facts_mod.warm_box_history("2026-03-18")
    first = facts_mod._box_score_history("2026-03-18")
    second = facts_mod._box_score_history("2026-03-18")

    assert counting_enrich["n"] == 1, f"the window was built {counting_enrich['n']} times"
    assert first is second, "the cached frame was not reused"


def test_two_game_dates_are_cached_side_by_side(offline, counting_enrich):
    """A week of upcoming games must not thrash the cache.

    One entry keyed on a single `as_of` rebuilt for every date in a pass, so a
    page-load across the week paid for a window per game and reused nothing.
    """
    _schedule_with_history(offline, DATES)

    dates = ["2026-03-18", "2026-03-19", "2026-03-20"]
    for date in dates:
        facts_mod.warm_box_history(date)
    assert counting_enrich["n"] == len(dates), (
        f"expected one build per distinct as_of, got {counting_enrich['n']}"
    )

    for date in dates:
        assert len(facts_mod._box_score_history(date)) == len(DATES), (
            f"{date} went cold between two reads of the same pass"
        )
    assert counting_enrich["n"] == len(dates), (
        f"the second pass rebuilt {counting_enrich['n'] - len(dates)} windows: the "
        "cache does not hold more than one as_of"
    )


def test_the_cache_is_bounded(offline, counting_enrich):
    """Bounded, so a month of dates cannot grow it without limit."""
    _schedule_with_history(offline, DATES)

    for i in range(facts_mod.BOX_HISTORY_KEEP + 6):
        facts_mod.warm_box_history(f"2026-04-{i + 1:02d}")

    assert len(facts_mod._BOX_HISTORY) == facts_mod.BOX_HISTORY_KEEP, (
        f"the cache grew to {len(facts_mod._BOX_HISTORY)} entries; it is unbounded"
    )


# --- refresh --------------------------------------------------------------

def test_a_changed_schedule_rebuilds_the_window(offline, monkeypatch, counting_enrich):
    """A game completing is the moment the duels' answer changes, so the cache has
    to notice. Ranking last month's form as if it were now is the failure here."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "schedule.json"
        path.write_text("[]")
        monkeypatch.setattr(facts_mod.deps, "get_schedule_path", lambda: path)
        monkeypatch.setattr(
            facts_mod, "_schedule",
            lambda: [_game(f"g{i}", d, "BOS", "MIA") for i, d in enumerate(DATES)],
        )

        facts_mod.warm_box_history("2026-03-18")
        assert counting_enrich["n"] == 1

        # The schedule cache gains a game. Same mtime, but its size moves.
        path.write_text('["a completed game"]')
        facts_mod.warm_box_history("2026-03-18")

        assert counting_enrich["n"] == 2, (
            "the schedule cache changed and the window was not rebuilt: the duels "
            "would keep quoting the form they were built with"
        )


def test_a_cold_window_after_a_schedule_change_is_not_served(offline, monkeypatch, counting_enrich):
    """The cache is keyed on the schedule state, so a changed schedule invalidates
    the entry it was holding rather than serving last month's form."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "schedule.json"
        path.write_text("[]")
        monkeypatch.setattr(facts_mod.deps, "get_schedule_path", lambda: path)
        monkeypatch.setattr(
            facts_mod, "_schedule",
            lambda: [_game(f"g{i}", d, "BOS", "MIA") for i, d in enumerate(DATES)],
        )

        facts_mod.warm_box_history("2026-03-18")
        path.write_text('["a completed game"]')      # the schedule moved on

        assert len(facts_mod._box_score_history("2026-03-18")) == 0, (
            "a stale window was served after the schedule changed"
        )


# --- the warmer -----------------------------------------------------------

def test_a_failed_warm_is_reported_and_does_not_raise(offline, monkeypatch):
    """A warmer that cannot read box scores must not take the app down."""
    def boom(as_of):
        raise RuntimeError("espn unavailable")

    monkeypatch.setattr(facts_mod, "_build_box_history", boom)

    assert facts_mod.warm_box_history("2026-03-18") is False


def test_the_warmer_warms_every_upcoming_date(offline, monkeypatch):
    """Every upcoming date, not the next one: a skipped date has no duels."""
    warmed: list[str] = []
    monkeypatch.setattr(
        facts_mod, "_schedule",
        lambda: [
            _game("a", "2026-03-18", "BOS", "MIA", completed=False),
            _game("b", "2026-03-19", "BOS", "MIA", completed=False),
            _game("c", "2026-03-19", "LAL", "BOS", completed=False),
            _game("d", "2026-03-20", "BOS", "MIA", completed=False),
        ],
    )
    monkeypatch.setattr(facts_mod, "warm_box_history", lambda as_of: warmed.append(as_of) or True)

    ticked = {"n": 0}

    facts_mod.start_box_history_warmer(
        __import__("pathlib").Path("/nonexistent/schedule.json"),
        sleep=lambda _: ticked.__setitem__("n", ticked["n"] + 1),
        should_stop=lambda: ticked["n"] >= 1,
    ).join(timeout=5)

    assert sorted(warmed) == ["2026-03-18", "2026-03-19", "2026-03-20"], (
        f"the warmer warmed {warmed}; every upcoming date needs a window"
    )


def test_the_lifespan_starts_the_warmer():
    """Serving must not depend on it, and it must be wired in."""
    import inspect

    assert "start_box_history_warmer" in inspect.getsource(app_mod.lifespan), (
        "the lifespan does not start the warmer"
    )
