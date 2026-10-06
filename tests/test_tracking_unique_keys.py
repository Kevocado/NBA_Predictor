# tests/test_tracking_unique_keys.py
"""The table-by-table audit, written down so it cannot quietly become wrong.

Six tables, and they do NOT get the same answer. The one below that looks
like an oversight -- four tables with no constraint on their business key -- is
the product working as specified, and this file is where that is pinned.

| table | declared row key | enforced? | duplicates possible? | can a duplicate move a graded aggregate? |
|---|---|---|---|---|
| `predictions` | `(game_id)` | no (surrogate `id` only) | yes, by design | no -- `counted_picks` -> `earliest_recorded` |
| `player_prediction_snapshots` | `(game_id, player_id, stat)` | no | yes, by design | no -- `resolved_player_props` -> `earliest_recorded` |
| `game_market_predictions` | one odds RUN | no | yes, by construction | no -- `_counted_market_pick` -> `earliest_run` |
| `game_forecast_snapshots` | none | no | yes | nothing reads it |
| `odds_timing_snapshots` | none | no | yes | nothing reads it |
| `game_player_outcomes` | `(game_id, player_id, stat)` | **yes** | not any more | was yes, before the index |

"By design" is not a hand-wave, it is two things this file checks: rule 2 of
the 2026-10-01 spec keeps a rerun as history that the published per-pick list
still shows, and the scheduled refresh re-presents a rolling window of finished
games every single day. A UNIQUE index on any of those three would not harden
them. It would abort the daily ingest and delete the history the rule reads.
That is why the constraint went on the one table whose key is a fact instead,
where a duplicate was a genuine silent corruption
(`tests/test_tracking_outcome_uniqueness.py`).
"""
import re
import sqlite3
from pathlib import Path

import pytest

from nba_predictor.tracking import store

REPO = Path(__file__).resolve().parents[1]
_PREDICTION_TABLES = ("predictions", "game_market_predictions", "player_prediction_snapshots")

# Prediction-shaped too, for the same reason: the odds refresher runs on a loop
# and re-snapshots each pick as often as the cadence allows, so a unique index
# would make the loop fail the moment it did its job. `value_ledger.read_ledger`
# dedupes to the earliest snapshot, exactly as `timing.earliest_recorded` does
# for the game-level tables.
_PREDICTION_TABLES += ("value_picks",)

# Fact tables: one row per thing that happened, so a duplicate would move a
# number. `game_player_outcomes` was the original; the value ledger's outcome
# and closing-line tables joined it under the same reasoning.
_FACT_TABLES = ("game_player_outcomes", "value_pick_outcomes", "value_closing_lines")


def _unique_indexed_columns(db_path):
    """`{table: {index_name: columns}}` for every unique index in the file."""
    out = {}
    with store.get_connection(db_path) as conn:
        for (table,) in conn.execute(
            "SELECT DISTINCT tbl_name FROM sqlite_master WHERE type='index' AND sql LIKE '%UNIQUE%'"
        ):
            for (name,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql LIKE '%UNIQUE%'",
                (table,),
            ):
                cols = [row[2] for row in conn.execute(f"PRAGMA index_info({name})")]
                out.setdefault(table, {})[name] = cols
    return out


def test_the_surrogate_primary_key_constrains_no_business_key(tmp_path):
    """What `id INTEGER PRIMARY KEY AUTOINCREMENT` actually promises.

    It is a PRIMARY KEY, so the original report that the schema has "no
    PRIMARY KEY, no UNIQUE" is not quite right -- but a surrogate one makes
    each ROW distinct and constrains no key a reader can name. Stated here as a
    test because the distinction is the whole finding: a reader who looked only
    for the words PRIMARY KEY would conclude this table was protected.
    """
    db = tmp_path / "tracking.db"
    store.init_db(db)

    with sqlite3.connect(db) as conn:
        for table in ("predictions", "player_prediction_snapshots", "game_player_outcomes"):
            info = conn.execute(f"PRAGMA table_info({table})").fetchall()
            pk = [row[1] for row in info if row[5]]
            assert pk == ["id"], f"{table}'s PRIMARY KEY is no longer the surrogate id"
        # `table_info` carries no uniqueness flag, so uniqueness is read from the
        # index list -- which is why the claim below needs a test rather than a
        # glance at the DDL.
        for table in ("predictions", "player_prediction_snapshots"):
            indexed = [row[1] for row in conn.execute(f"PRAGMA index_list({table})") if row[2]]
            assert not indexed, (
                f"{table} gained a unique index ({indexed}); re-run the audit -- "
                "store._UNIQUE_INDEXES says it should not have"
            )
        # The one exception, and it is an index on the ROW KEY, not on `id`.
        outcomes_unique = [row[1] for row in conn.execute(
            "PRAGMA index_list(game_player_outcomes)") if row[2]]
        assert outcomes_unique == ["uq_game_player_outcomes_key"]
        assert [
            row[2] for row in conn.execute("PRAGMA index_info(uq_game_player_outcomes_key)")
        ] == ["game_id", "player_id", "stat"]

    # Two rows, same game, both accepted: the key a reader cares about is free.
    store.insert_prediction(
        db, game_id="g1", created_at="2026-09-01T09:00:00+00:00", model_version="v1",
        home_win_prob=0.9, predicted_margin=1.0, predicted_total=220.0,
    )
    store.insert_prediction(
        db, game_id="g1", created_at="2026-09-01T10:00:00+00:00", model_version="v2",
        home_win_prob=0.1, predicted_margin=1.0, predicted_total=220.0,
    )
    assert len(store.get_predictions_for_game(db, "g1")) == 2


def test_the_three_prediction_tables_carry_no_unique_index_and_that_is_chosen(tmp_path):
    """The audit's negative result, asserted so a future change is deliberate.

    If someone adds a unique index to any of these three, this fails and the
    reasoning has to be revisited -- because the daily refresh
    (`refresh-data.yml` -> `pipeline/ingest.py`) scores a rolling 60-day window
    of finished games on every run, so the duplicate below is written every
    single day in production. It is not a mistake to be prevented; it is rule 2
    of the 2026-10-01 spec, kept as history and counted once by the reader.
    """
    db = tmp_path / "tracking.db"
    store.init_db(db)

    indexed = _unique_indexed_columns(db)

    assert set(indexed) == set(_FACT_TABLES), (
        f"unique index coverage changed to {sorted(indexed)}; re-check the "
        f"table-by-table audit before landing. Expected exactly the fact "
        f"tables {sorted(_FACT_TABLES)}."
    )
    for table in _PREDICTION_TABLES:
        assert table not in indexed

    # The duplicate the daily refresh really writes: same game, later stamp.
    for created_at in ("2026-09-01T09:00:00+00:00", "2026-09-01T10:00:00+00:00"):
        store.insert_prediction(
            db, game_id="g1", created_at=created_at, model_version="v1",
            home_win_prob=0.9, predicted_margin=1.0, predicted_total=220.0,
        )
    assert len(store.get_predictions_for_game(db, "g1")) == 2, (
        "a rerun must still land twice; it is kept as history and counted once "
        "by the reader, not blocked by the schema"
    )


def test_the_two_tables_nothing_reads_have_been_left_alone(tmp_path):
    """`game_forecast_snapshots` and `odds_timing_snapshots` get no invented key.

    Neither has a writer or a reader anywhere in `src/`, so there is no
    aggregate to protect. Giving them a key would be a claim about a table
    nothing uses, and the constraint would be unverifiable against real data.
    """
    source = "\n".join(
        path.read_text() for path in (REPO / "src").rglob("*.py")
    )
    for table in ("game_forecast_snapshots", "odds_timing_snapshots"):
        # Only store.py's own SCHEMA may name them.
        others = [
            path for path in (REPO / "src").rglob("*.py")
            if table in path.read_text() and path.name != "store.py"
        ]
        assert not others, (
            f"{table} now has code outside store.py ({[p.name for p in others]}); "
            "it may need a row key -- re-run the audit"
        )
        assert source.count(table) >= 1  # still declared in SCHEMA

    db = tmp_path / "tracking.db"
    store.init_db(db)
    # The only indexed tables are the fact tables. `value_picks` is deliberately
    # absent: the odds refresher re-snapshots on a loop, so indexing it would
    # break the loop rather than protect a number.
    assert set(_unique_indexed_columns(db)) == set(_FACT_TABLES)


def test_the_write_path_is_insert_only_so_the_new_constraint_is_the_whole_guard(tmp_path):
    """No writer can rewrite an outcome to satisfy a constraint after the fact.

    The reader half of the fix assumes outcomes are immutable. `store.py` is
    the only module that writes them, and it writes with INSERT only -- the
    sibling invariant in `test_track_record_counted_picks.py` already pins that
    for `created_at`, and it now also holds for `actual_value`: there is no
    UPDATE path a future migration could take to reconcile a conflict by
    rewriting a recorded pick.
    """
    source = Path(store.__file__).read_text()
    code = re.sub(r'""".*?"""', "", source, flags=re.S)

    assert not re.search(r"\bUPDATE\b", code, re.I)
    assert not re.search(r"\bDELETE\b", code, re.I)
    assert "OR IGNORE" not in code and "OR REPLACE" not in code, (
        "an INSERT OR IGNORE/REPLACE would now silently drop or overwrite a row "
        "against the new constraint; that is the degradation this PR closes"
    )


@pytest.mark.parametrize("table", _PREDICTION_TABLES + ("game_player_outcomes",))
def test_every_prediction_and_outcome_table_is_still_declared_in_schema(table):
    """`test_track_record_counted_picks.py` parses SCHEMA for the tables it
    checks; this keeps the parse targets all six and fails loudly if one is
    renamed or dropped."""
    assert re.search(rf"CREATE TABLE IF NOT EXISTS {table} \(", store.SCHEMA)