"""The multi-season backfill, tested without touching the network.

Two claims carry the weight here and both are pinned:

  * a backfill fills the cache and writes the training cache, and does NOT
    retrain or write to the tracking database. That is what separates it from
    `run_ingest`, which would write ~6,000 predictions into the record the site
    publishes;
  * the throttle really paces requests, because a backfill that fires
    thousands of calls at full speed is how it gets throttled into 429s.

Every fetch is injected. Nothing here opens a socket, and one test asserts that
of the whole module.
"""

import json
from datetime import date

import pytest

from nba_predictor import config
from nba_predictor.data import espn
from nba_predictor.pipeline import backfill


def _game(game_id: str, game_date: str, *, completed=True, with_box=True) -> dict:
    g = {
        "game_id": game_id, "game_date": game_date, "tip_off": f"{game_date}T00:00Z",
        "home_team": "BOS", "away_team": "LAL", "completed": completed,
        "home_pts": 110 if completed else None, "away_pts": 100 if completed else None,
    }
    if with_box and completed:
        for side in ("home", "away"):
            g[f"{side}_fgm"] = 40
            g[f"{side}_fga"] = 88
            g[f"{side}_fg3m"] = 12
            g[f"{side}_tov"] = 11
            g[f"{side}_oreb"] = 9
            g[f"{side}_dreb"] = 32
            g[f"{side}_fta"] = 20
    return g


def test_season_bounds_cover_one_season_from_october_to_april():
    start, end = backfill.season_bounds(2023)
    assert start == "2023-10-20"
    assert end == "2024-04-15"


def test_prior_seasons_walks_backwards_and_stops_before_the_current_one():
    # Oct 2026 is inside the 2026 season, so "the season before" starts 2025.
    assert backfill.prior_seasons(3, before=date(2026, 10, 7)) == [2023, 2024, 2025]
    # January belongs to the season that started the year before.
    assert backfill.prior_seasons(2, before=date(2026, 1, 15)) == [2023, 2024]


def test_backfill_writes_a_training_cache_and_asks_for_each_season(tmp_path):
    asked: list[tuple[str, str]] = []

    def fetch(start, end):
        asked.append((start, end))
        return [_game("g1", start), _game("g2", start)]

    def enrich(games):
        return games

    out = backfill.backfill_seasons(
        [2023, 2024], training_path=tmp_path / "games.json",
        fetch=fetch, enrich=enrich, log=lambda _m: None,
    )

    assert asked == [("2023-10-20", "2024-04-15"), ("2024-10-20", "2025-04-15")]
    assert out["n_training_rows"] == 4
    written = json.loads((tmp_path / "games.json").read_text())
    assert len(written) == 4
    assert written[0]["game_id"] == "g1"


def _imported_and_called_names(module) -> tuple[set[str], set[str]]:
    """What the module actually imports and calls.

    Parsed rather than grepped, because the module's own docstrings name
    `run_ingest` and `retrain` while explaining that it does neither -- and a
    substring check over the source would match that prose and fail on a file
    that is telling the truth.
    """
    import ast

    tree = ast.parse(open(module.__file__).read())
    imported: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
            imported.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                called.add(f.id)
            elif isinstance(f, ast.Attribute):
                called.add(f.attr)
    return imported, called


def test_backfill_does_not_train_and_does_not_touch_the_tracking_db():
    """The distinction from run_ingest, stated as a test.

    A five-season backfill through `run_ingest` would train a model and store a
    prediction for every historical game, which is a backtest wearing a
    backfill's clothes: it would rewrite the record the site publishes, using
    outcomes it now already knows.
    """
    _, called = _imported_and_called_names(backfill)

    for forbidden in ("retrain", "run_retrain_pipeline", "insert_prediction",
                      "score_and_store_predictions", "run_ingest", "insert_player_outcome"):
        assert forbidden not in called, f"the backfill calls {forbidden}"

    # And it takes no db_path at all, so there is nothing to write through.
    import inspect

    assert "db_path" not in inspect.signature(backfill.backfill_seasons).parameters


def test_a_game_without_a_box_score_is_not_counted_as_a_training_row(tmp_path):
    def fetch(start, end):
        return [
            _game("with-box", "2024-01-05"),
            _game("no-box", "2024-01-06", with_box=False),
            _game("unplayed", "2024-01-07", completed=False),
        ]

    out = backfill.backfill_seasons(
        [2023], training_path=tmp_path / "games.json",
        fetch=fetch, enrich=lambda g: g, log=lambda _m: None,
    )

    assert out["n_training_rows"] == 1, "a game with no box score became a training row"
    assert json.loads((tmp_path / "games.json").read_text())[0]["game_id"] == "with-box"


def test_season_summaries_report_what_each_season_actually_contained(tmp_path):
    def fetch(start, end):
        return [_game("g1", start)]

    out = backfill.backfill_seasons(
        [2022, 2023], training_path=tmp_path / "games.json",
        fetch=fetch, enrich=lambda g: g, log=lambda _m: None,
    )

    assert set(out["seasons"]) == {"2022", "2023"}
    for stats in out["seasons"].values():
        assert stats["games_found"] == 1
        assert stats["games_with_box"] == 1
        assert stats["start"] < stats["end"]


# --- the throttle ---------------------------------------------------------

def test_the_throttle_paces_requests_by_the_configured_interval(monkeypatch):
    """It must actually wait between calls, not merely record a timestamp."""
    slept: list[float] = []
    clock_value = [0.0]

    def clock() -> float:
        return clock_value[0]

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock_value[0] += seconds  # time passes because we slept

    monkeypatch.setattr(espn, "ESPN_MIN_REQUEST_INTERVAL_SECONDS", 0.5)
    monkeypatch.setattr(espn, "_last_request_at", None)

    espn._throttle(sleep=sleep, clock=clock)          # first call: nothing to wait for
    clock_value[0] += 0.1                            # 0.1s since the last request
    espn._throttle(sleep=sleep, clock=clock)          # must wait 0.4s

    assert slept == [pytest.approx(0.4)], "the second request was not paced"


def test_the_throttle_never_waits_when_the_interval_has_elapsed(monkeypatch):
    slept: list[float] = []
    clock_value = [0.0]

    monkeypatch.setattr(espn, "ESPN_MIN_REQUEST_INTERVAL_SECONDS", 0.2)
    monkeypatch.setattr(espn, "_last_request_at", None)

    espn._throttle(sleep=slept.append, clock=lambda: clock_value[0])
    clock_value[0] += 5.0                             # plenty of time
    espn._throttle(sleep=slept.append, clock=lambda: clock_value[0])

    assert slept == [], "it slept although the interval had long passed"


def test_the_throttle_is_off_when_the_interval_is_zero(monkeypatch):
    """A fixture run that must not sleep sets the interval to 0."""
    slept: list[float] = []
    monkeypatch.setattr(espn, "ESPN_MIN_REQUEST_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(espn, "_last_request_at", None)

    espn._throttle(sleep=slept.append, clock=lambda: 0.0)
    espn._throttle(sleep=slept.append, clock=lambda: 0.0)

    assert slept == []


def test_the_backfill_module_makes_no_network_call_of_its_own():
    """It composes `fetch_schedule_range` and `enrich_with_boxscores`; it does
    not import `requests` or reach ESPN itself."""
    imported, _ = _imported_and_called_names(backfill)

    assert "requests" not in imported, "the backfill imports a network client"
    assert "espn" not in imported, "the backfill reaches for ESPN directly"
    # Its only route to the network is the two injected pipeline stages.
    assert {"fetch_schedule_range", "enrich_with_boxscores"} <= imported

def test_one_season_failing_does_not_abandon_the_others(tmp_path):
    """A multi-hour fetch will lose the network. It has to keep what it got.

    This is not hypothetical: the first real run fetched a whole season
    (1,238 of 1,240 games) and then died on a DNS failure inside the second,
    throwing away every game it had because it had not written its file yet.
    """
    def fetch(start, end):
        if start.startswith("2024"):
            raise ConnectionError("DNS went away")
        return [_game("g1", start)]

    out = backfill.backfill_seasons(
        [2023, 2024, 2025], training_path=tmp_path / "games.json",
        fetch=fetch, enrich=lambda g: g, log=lambda _m: None,
    )

    # The two good seasons still landed.
    assert out["n_training_rows"] == 2
    assert out["failed_seasons"] == ["2024"]
    assert "games_with_box" not in out["seasons"]["2024"]
    assert "failed" in out["seasons"]["2024"]
    # A partial run must be visible as partial, not mistaken for a complete one.
    assert out["failed_seasons"] == ["2024"]


def test_a_complete_run_reports_no_failed_seasons(tmp_path):
    out = backfill.backfill_seasons(
        [2023], training_path=tmp_path / "games.json",
        fetch=lambda s, e: [_game("g1", s)], enrich=lambda g: g, log=lambda _m: None,
    )
    assert out["failed_seasons"] == []
