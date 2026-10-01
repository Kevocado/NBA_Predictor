"""NEUTRALISED. This module used to fabricate injury data and now refuses.

What it did, on main up to 2026-09-30: get_current_injuries() returned a
hardcoded list containing a literal injury for a named real player ("LeBron
James", Questionable, "Out with right knee soreness"), get_player_injury_history
returned a hardcoded "Ankle sprain" dated 2023-01-15, and _fetch_player_stats
returned {} -- all behind a module docstring reading "Uses nbainjuries / NBA
official injury report with caching and retry logic", and a "# Fetch from API
(using sample data as placeholder)" comment where the fetch should have been.

It was not dead code either. nba_predictor/data/__init__.py imported it, so
importing the data package reached the fabrications, and
features/injuries.py called get_missing_player_value by default, summing a
fabricated 0.0 for every missing player -- a number that reads exactly like a
measured zero. Any production path that had used this module would have
reported an injury that does not exist, attributed to a real person.

Every entry point now raises NotImplementedError. It is kept, rather than
deleted, so the import in features/injuries.py and data/__init__.py resolves
and a future caller fails loudly at the call instead of silently importing a
fake. tests/test_data_injuries.py pins the refusal and asserts the fabricated
strings are gone from this source.

THE REAL SOURCE for availability is nba_predictor.data.espn.get_injuries(),
which fetches ESPN's live report and resolves each row to the same ESPN
athlete id the prop rows are keyed by.
"""


def _refuse(name: str):
    raise NotImplementedError(
        f"data/injuries.py.{name} is neutralised: it returned fabricated sample data "
        "(a hardcoded injury for a named real player), not a real injury report. "
        "Use nba_predictor.data.espn.get_injuries(), which reads ESPN's live report."
    )


def get_current_injuries() -> list[dict]:
    _refuse("get_current_injuries")


def get_player_injury_history(player_id: int) -> list[dict]:
    _refuse("get_player_injury_history")


def get_missing_player_value(player_id: int, season: int) -> float:
    _refuse("get_missing_player_value")
