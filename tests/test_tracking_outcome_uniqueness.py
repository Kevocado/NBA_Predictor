# tests/test_tracking_outcome_uniqueness.py
"""`game_player_outcomes` enforces its row key, and the reader keys on the same one.

The defect this closes, in one line: of the six tables in `store.SCHEMA`, the
four whose row key is a PICK key declare no constraint at all, and the one
whose row key is a FACT -- `game_player_outcomes`, the table every player-prop
MAE is graded against -- declared no constraint either. Its `id INTEGER PRIMARY
KEY AUTOINCREMENT` is a PRIMARY KEY, but a surrogate one: it makes each row
distinct and constrains no business key, so two rows could claim the same
`(game_id, player_id, stat)` and the read side had no dedupe and no ORDER BY.
The MAE was then computed against whichever row SQLite happened to hand back
last. A number moves and nothing records why.

The four prediction tables keep their duplicates ON PURPOSE. Rule 2 of the
2026-10-01 spec (predictor-hub #66) keeps a rerun as history -- still in the
table, still in the published per-pick list, counted once by `earliest_recorded`
-- and the scheduled refresh re-presents a rolling window of finished games
every day. `tests/test_tracking_unique_keys.py` pins that reading, including
that a UNIQUE index on those tables would abort the daily ingest.

So: the constraint goes where a duplicate is a bug, and the reader dedupe goes
everywhere, because the two are different jobs. The index stops a duplicate
being written; `earliest_recorded_outcome` makes a file that predates the index
grade deterministically instead of arbitrarily. Neither is a substitute for the
other, and they agree on ONE key: `(game_id, player_id, stat)`, defined once in
`timing._counting_key` and once in `store._UNIQUE_INDEXES`.
"""
import sqlite3

import pytest

from nba_predictor.tracking import store
from nba_predictor.tracking import timing
from nba_predictor.tracking.player_props import resolved_player_props

# `store.ConflictingOutcome` is referenced through the module rather than
# imported by name, deliberately: on a checkout without it, every test in this
# file fails on its own assertion instead of the whole module erroring at
# collection, so a red-check against `main` reports what broke rather than one
# opaque ImportError.

_INDEX = "uq_game_player_outcomes_key"


def _completed(game_id, home, away, home_pts, away_pts, **over):
    return {
        "game_id": game_id, "game_date": "2026-03-01", "home_team": home, "away_team": away,
        "completed": True, "home_pts": home_pts, "away_pts": away_pts, **over,
    }


def _db(tmp_path):
    db = tmp_path / "tracking.db"
    store.init_db(db)
    return db


def _outcome(db, game_id="g1", player_id="p1", stat="points", value=24.0, at="2026-03-02T09:00:00+00:00"):
    return store.insert_player_outcome(
        db, game_id=game_id, player_id=player_id, stat=stat,
        actual_value=value, recorded_at=at,
    )


def _prediction(db, game_id="g1", player_id="p1", stat="points", value=20.0, at="2026-03-01T09:00:00+00:00"):
    return store.insert_player_prediction(
        db, game_id=game_id, player_id=player_id, stat=stat,
        predicted_value=value, created_at=at,
    )


def _legacy_db_with_duplicates(tmp_path):
    """A database shaped like one written BEFORE the index existed.

    Built by dropping the index rather than by weakening the writer, so the
    duplicate rows are genuinely sitting in the table the way a pre-change file
    has them -- same schema, same columns, no constraint covering them.
    """
    db = _db(tmp_path)
    with store.get_connection(db) as conn:
        # `IF EXISTS` so that on a checkout WITHOUT the index this fixture still
        # builds the duplicate rows. That is what makes the red-check honest:
        # against `main` the test then fails on its own assertion -- the MAE was
        # graded against the wrong row -- rather than dying in the setup.
        conn.execute(f"DROP INDEX IF EXISTS {_INDEX}")
        conn.commit()
    # No index, so both land. The second disagrees, which is the dangerous shape:
    # two rows, one key, two different actual values, nothing to choose between
    # them but SQLite's scan order.
    with store.get_connection(db) as conn:
        for value, at in ((24.0, "2026-03-02T09:00:00+00:00"), (48.0, "2026-03-09T09:00:00+00:00")):
            conn.execute(
                "INSERT INTO game_player_outcomes (game_id, player_id, stat, actual_value, recorded_at)"
                " VALUES ('g1', 'p1', 'points', ?, ?)", (value, at),
            )
        conn.commit()
    return db


# --- the schema rejects the duplicate -----------------------------------------

def test_a_duplicate_outcome_row_for_the_same_key_is_rejected_by_the_schema(tmp_path):
    """The constraint, stated the only way it can be tested: raw SQL.

    Deliberately NOT through `insert_player_outcome`, which now declines to
    write the row before the database ever sees it. A test that only exercised
    the writer would pass even with no index at all, and the whole point is that
    the SCHEMA is what holds.
    """
    db = _db(tmp_path)
    _outcome(db)

    with store.get_connection(db) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO game_player_outcomes"
                " (game_id, player_id, stat, actual_value, recorded_at)"
                " VALUES ('g1', 'p1', 'points', 48.0, '2026-03-09T09:00:00+00:00')"
            )

    assert len(store.get_player_outcomes_for_game(db, "g1")) == 1


def test_the_index_covers_exactly_the_key_the_reader_dedupes_on(tmp_path):
    """The reader's key and the constraint's key are the same key.

    A reader deduping on a key the schema does not cover is exactly the silent
    hole PL shipped (a dedupe on a `market` column the table did not have), and
    it is invisible from either side alone: the constraint can be right and the
    reader still wrong, or the reverse. So both halves are asserted here against
    each other, derived from the SQL rather than restated.
    """
    db = _db(tmp_path)
    _outcome(db, player_id="p1")

    with store.get_connection(db) as conn:
        indexed = {row[2] for row in conn.execute(f"PRAGMA index_info({_INDEX})")}
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='game_player_outcomes'"
            " AND sql LIKE '%UNIQUE%'"
        )}

    assert indexed == {"game_id", "player_id", "stat"}
    assert tables == {_INDEX}
    # The reader's key, read out of the same function the reader uses.
    assert timing._counting_key(
        {"game_id": "g1", "player_id": "p1", "stat": "points", "recorded_at": "x"}
    ) == ("g1", "p1", "points")


def test_a_non_duplicate_outcome_row_still_records(tmp_path):
    """The constraint must not cost the ability to record.

    Two players on one stat in one game, and the same player on two stats --
    every one of those is a distinct key, and all of them are legitimate
    settlement rows. A constraint on the wrong columns would have eaten one.
    """
    db = _db(tmp_path)
    _outcome(db, game_id="g1", player_id="p1", stat="points")
    _outcome(db, game_id="g1", player_id="p2", stat="points")
    _outcome(db, game_id="g1", player_id="p1", stat="rebounds")
    _outcome(db, game_id="g2", player_id="p1", stat="points")

    assert len(store.get_player_outcomes_for_game(db, "g1")) == 3
    assert len(store.get_player_outcomes_for_game(db, "g2")) == 1
    assert store.conflicting_outcome_keys(db) == []


# --- the writer --------------------------------------------------------------

def test_recording_the_same_outcome_twice_writes_one_row_and_keeps_the_first(tmp_path):
    """Why the writer needs a check of its own, and why it is not OR IGNORE.

    The scheduled refresh re-presents a rolling window of finished games on
    every run, so this is the ordinary case, not an edge case. The recorded row
    keeps its own id and its own `recorded_at` -- nothing is rewritten, which is
    rule 1 -- and no second row appears to be graded against.
    """
    db = _db(tmp_path)
    first = _outcome(db, value=24.0, at="2026-03-02T09:00:00+00:00")

    second = _outcome(db, value=24.0, at="2026-03-09T09:00:00+00:00")

    assert second == first
    rows = store.get_player_outcomes_for_game(db, "g1")
    assert len(rows) == 1
    assert rows[0]["actual_value"] == pytest.approx(24.0)
    assert rows[0]["recorded_at"] == "2026-03-02T09:00:00+00:00"


def test_a_disagreeing_outcome_is_refused_loudly_and_never_overwrites(tmp_path):
    """The silent-degradation question, answered for the writer.

    A second, DIFFERENT value for a key already recorded is a contradiction --
    the kind a revised box score produces. `INSERT OR IGNORE` would swallow it
    and `INSERT OR REPLACE` would rewrite an immutable pick, so neither is
    acceptable; the row is refused by name and the original stands.
    """
    db = _db(tmp_path)
    _outcome(db, value=24.0)

    with pytest.raises(store.ConflictingOutcome) as excinfo:
        _outcome(db, value=48.0, at="2026-03-09T09:00:00+00:00")

    assert "g1" in str(excinfo.value) and "48.0" in str(excinfo.value)
    rows = store.get_player_outcomes_for_game(db, "g1")
    assert len(rows) == 1
    assert rows[0]["actual_value"] == pytest.approx(24.0), "a recorded outcome was rewritten"


def test_the_ingest_reports_a_conflict_and_does_not_abort_the_run(tmp_path):
    """One revised box score must not stop the other 399 rows being recorded.

    This is the path that reaches `store_player_outcomes`: the refresh calls it
    over a whole window, so a conflict has to be counted and named rather than
    raised out of the loop.
    """
    import pandas as pd

    from nba_predictor.pipeline.ingest import store_player_outcomes

    db = _db(tmp_path)
    frame = pd.DataFrame([
        {"game_id": "g1", "player_id": "p1", "points": 48.0, "rebounds": 7.0, "assists": 4.0, "fg3m": 2.0},
        {"game_id": "g1", "player_id": "p2", "points": 12.0, "rebounds": 3.0, "assists": 9.0, "fg3m": 0.0},
    ])
    _outcome(db, game_id="g1", player_id="p1", stat="points", value=24.0)

    logged = []
    stored = store_player_outcomes(frame, db, log=logged.append)

    # p1's points conflict; its other three stats are new, and all four of p2's.
    assert stored == 7, "a conflict must not take the rest of the window down with it"
    assert len(logged) == 1 and "disagreed" in logged[0]
    assert store.get_player_outcomes_for_game(db, "g1")[0]["actual_value"] == pytest.approx(24.0)


def test_a_second_pass_over_the_same_window_adds_nothing_and_reports_zero(tmp_path):
    """The count is facts ADDED, not calls made.

    The daily refresh walks a rolling window, so the second pass presents
    almost everything already stored. Reporting those as new rows is how a
    summary quietly starts lying about what a run did.
    """
    import pandas as pd

    from nba_predictor.pipeline.ingest import store_player_outcomes

    db = _db(tmp_path)
    frame = pd.DataFrame([
        {"game_id": "g1", "player_id": "p1", "points": 24.0, "rebounds": 7.0, "assists": 4.0, "fg3m": 2.0},
    ])

    assert store_player_outcomes(frame, db, log=lambda _: None) == 4
    assert store_player_outcomes(frame, db, log=lambda _: None) == 0
    assert store.conflicting_outcome_keys(db) == []


# --- the reader still dedupes ------------------------------------------------

def test_the_reader_keeps_the_earliest_recorded_row_for_a_duplicated_key(tmp_path):
    """The reader is defence in depth, and it must work on a file with no index.

    Before the fix this read folded rows into a dict with no dedupe and no
    ORDER BY, so with 24.0 recorded first and 48.0 six days later the MAE was
    graded against whichever row SQLite scanned last -- the error term doubled
    or halved and nothing said why. The reader now applies the same
    earliest-recorded rule every other counted pick in this repo gets.
    """
    db = _legacy_db_with_duplicates(tmp_path)
    _prediction(db, value=20.0)
    schedule = [_completed("g1", "BOS", "MIA", 110, 100)]

    rows = resolved_player_props(db, schedule)

    assert len(rows) == 1
    assert rows[0]["actual_value"] == pytest.approx(24.0), (
        "the counted outcome is the earliest recorded one, not the last scanned"
    )
    # ...and the MAE that serving publishes follows it.
    from nba_predictor.models.player_props import in_sample_mae_by_stat
    assert in_sample_mae_by_stat(rows)["points"] == pytest.approx(4.0)


def test_the_reader_is_independent_of_the_order_the_rows_come_back_in(tmp_path):
    """Scan order is not a rule, so the answer must not depend on it.

    Both rows are read in one query, and SQLite makes no promise about which
    comes first. Feeding the reader the same two rows in the opposite order --
    and re-inserting them in the opposite order -- has to give the same number,
    or the fix only holds for whichever order the planner happened to pick.
    """
    db = _legacy_db_with_duplicates(tmp_path)
    _prediction(db, value=20.0)
    schedule = [_completed("g1", "BOS", "MIA", 110, 100)]

    forwards = resolved_player_props(db, schedule)

    rows = store.get_player_outcomes_for_game(db, "g1")
    backwards = timing.earliest_recorded_outcome(list(reversed(rows)))

    assert forwards[0]["actual_value"] == pytest.approx(24.0)
    assert [r["actual_value"] for r in backwards] == [pytest.approx(24.0)]


def test_the_reader_dedupe_does_not_change_a_clean_database(tmp_path):
    """The constraint must not change what serving reads.

    On a database this version creates there is exactly one row per key, so the
    reader's dedupe is the identity on the data serving actually sees. If this
    ever fails, the "add the index, change nothing" claim is no longer true.
    """
    db = _db(tmp_path)
    _prediction(db, value=20.0)
    for player_id, predicted, actual in (("p1", 20.0, 24.0), ("p2", 30.0, 27.0)):
        _prediction(db, player_id=player_id, value=predicted)
        _outcome(db, player_id=player_id, value=actual)
    schedule = [_completed("g1", "BOS", "MIA", 110, 100)]

    rows = resolved_player_props(db, schedule)

    assert sorted((r["player_id"], r["predicted_value"], r["actual_value"]) for r in rows) == [
        ("p1", 20.0, 24.0), ("p2", 30.0, 27.0),
    ]


# --- migration safety --------------------------------------------------------

def test_init_db_on_a_file_that_already_holds_a_conflicting_row_does_not_raise(tmp_path):
    """`init_db` runs on EVERY boot, so a migration it cannot finish must not
    be a migration that takes the service down.

    `CREATE UNIQUE INDEX` refuses to build over existing conflicts, and rule 1
    forbids deleting the rows that block it. The rows are left alone, the index
    is reported as not in place, and the reader carries the dedupe -- loudly,
    because a constraint that looks enforced and is not is worse than none.
    """
    db = _legacy_db_with_duplicates(tmp_path)

    with pytest.warns(RuntimeWarning, match="uq_game_player_outcomes_key"):
        store.init_db(db)

    # Nothing was deleted or rewritten: both rows are still there, as recorded.
    rows = store.get_player_outcomes_for_game(db, "g1")
    assert sorted(r["actual_value"] for r in rows) == [24.0, 48.0]
    assert store.conflicting_outcome_keys(db) == [("g1", "p1", "points", 2)]
    # ...and the reader still grades one pick, deterministically.
    _prediction(db, value=20.0)
    schedule = [_completed("g1", "BOS", "MIA", 110, 100)]
    assert resolved_player_props(db, schedule)[0]["actual_value"] == pytest.approx(24.0)


def test_init_db_builds_the_index_on_an_empty_table_and_is_idempotent(tmp_path):
    """The normal case: no rows to conflict with, so it lands, and a second
    call is a no-op rather than an error."""
    db = _db(tmp_path)
    store.init_db(db)
    store.init_db(db)

    with store.get_connection(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=?", (_INDEX,)
        ).fetchone()[0] == 1


def test_the_committed_tracking_database_has_no_duplicated_outcome_key():
    """Reported on the shipped data, not assumed.

    `data/tracking.db` is not committed, so this skips -- but if a future
    snapshot ships one, whoever lands it has to say whether the constraint could
    be built over it, which is the migration-safety question in its rawest form.
    """
    from pathlib import Path

    db = Path(__file__).resolve().parents[1] / "data" / "tracking.db"
    if not db.exists():
        pytest.skip("no committed tracking database in this checkout")
    store.init_db(db)
    assert store.conflicting_outcome_keys(db) == []