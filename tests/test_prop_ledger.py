"""Per-player prop ledger: writers and readers for tables that already exist.

Spec section 7. `player_prediction_snapshots` and `game_player_outcomes` have
been in the schema all along with nothing reading or writing them, so the site
has no per-player graded record — the prop equivalent of the game-level
track record the honesty machinery already maintains.

Three properties are load-bearing, and each is a way this could quietly lie:

1. **Snapshots are immutable.** A prop prediction snapshotted before tip-off is
   a record of what we published and when. If a later run overwrites it, the
   row stops being evidence and becomes a copy of the latest opinion. Re-running
   the pipeline must not displace the earliest pick — the same rule the
   game-level track record already follows.

2. **The player join never guesses.** ESPN identifies players by
   `player_id`; the sportsbook props feed identifies them by name. Joining them
   by fuzzy name match is how a real player gets someone else's line. The
   normaliser below is exact-match on (name, team); anything unmatched is
   logged and skipped.

3. **Position is recorded at snapshot time**, not looked up later. A player's
   position can change between the pick and the settlement, and reading it
   afterwards would attribute the pick to a role they did not have.
"""

import sqlite3

import pytest

from nba_predictor.tracking import store
from nba_predictor.tracking.store import (
    join_player_ids,
    read_prop_ledger,
    write_prop_snapshot,
)


PRE_TIP = "2026-10-21T18:00:00+00:00"
PRE_TIP_LATER = "2026-10-21T19:30:00+00:00"


def _db(tmp_path) -> str:
    path = tmp_path / "tracking.db"
    store.init_db(path)
    return str(path)


def test_snapshot_roundtrips_with_position(tmp_path):
    db = _db(tmp_path)
    write_prop_snapshot(db, game_id="g1", player_id="p1", stat="points",
                       predicted_value=23.5, created_at=PRE_TIP, position="G")

    rows = read_prop_ledger(db, game_id="g1")
    assert len(rows) == 1
    assert rows[0]["position"] == "G"
    assert rows[0]["predicted_value"] == 23.5
    assert rows[0]["player_id"] == "p1"
    assert rows[0]["stat"] == "points"


def test_a_later_snapshot_does_not_displace_the_earliest(tmp_path):
    """The pick that counts is the one published first.

    Note how immutability is achieved: the table keeps BOTH snapshots as history
    and the *reader* dedupes. That is this repo's standing decision, pinned by
    `test_tracking_unique_keys.py`: the daily refresh legitimately re-scores a
    rolling window every day, so a repeated snapshot is not a defect and must not
    be blocked by the schema. Adding a unique index here would contradict that,
    so it is the read that protects the published line.
    """
    db = _db(tmp_path)
    write_prop_snapshot(db, game_id="g1", player_id="p1", stat="points",
                       predicted_value=23.5, created_at=PRE_TIP, position="G")
    write_prop_snapshot(db, game_id="g1", player_id="p1", stat="points",
                       predicted_value=31.0, created_at=PRE_TIP_LATER, position="G")

    # History is kept, not truncated.
    with store.get_connection(db) as conn:
        stored = conn.execute(
            "SELECT COUNT(*) FROM player_prediction_snapshots WHERE game_id='g1'"
        ).fetchone()[0]
    assert stored == 2, "history must be kept -- the schema is not the immutability boundary"

    rows = read_prop_ledger(db, game_id="g1")
    assert len(rows) == 1, "the ledger reported a rerun as a second pick"
    assert rows[0]["predicted_value"] == 23.5, "the earliest snapshot was displaced"
    assert rows[0]["created_at"] == PRE_TIP


def test_no_unique_index_is_added_to_the_snapshot_table(tmp_path):
    """The reader dedupes, so adding a unique index would contradict the
    repo's audited decision and silently break the daily refresh."""
    db = _db(tmp_path)
    with store.get_connection(db) as conn:
        indexed = [r[1] for r in conn.execute("PRAGMA index_list(player_prediction_snapshots)") if r[2]]
    assert indexed == [], f"a unique index appeared on player_prediction_snapshots: {indexed}"


def test_distinct_markets_for_one_player_are_distinct_picks(tmp_path):
    db = _db(tmp_path)
    for stat in ("points", "rebounds", "assists", "threes"):
        write_prop_snapshot(db, game_id="g1", player_id="p1", stat=stat,
                            predicted_value=10.0, created_at=PRE_TIP, position="G")
    assert len(read_prop_ledger(db, game_id="g1")) == 4


def test_ledger_is_gradeable_against_outcomes(tmp_path):
    """A snapshot with no recorded outcome is not graded; one with an outcome
    is. Ungradeable markets must read as None, never as a miss — the
    game-level track record already learned that lesson."""
    db = _db(tmp_path)
    write_prop_snapshot(db, game_id="g1", player_id="p1", stat="points",
                       predicted_value=23.5, created_at=PRE_TIP, position="G")
    write_prop_snapshot(db, game_id="g1", player_id="p1", stat="rebounds",
                        predicted_value=6.0, created_at=PRE_TIP, position="G")

    rows = read_prop_ledger(db, game_id="g1")
    ungraded = {r["stat"]: r for r in rows}
    assert ungraded["points"]["actual_value"] is None
    assert ungraded["points"]["hit"] is None, "an ungraded pick must not read as a miss"
    assert ungraded["points"]["error"] is None

    store.insert_player_outcome(db, game_id="g1", player_id="p1", stat="points",
                               actual_value=25.0, recorded_at="2026-10-22T02:00:00+00:00")

    rows = {r["stat"]: r for r in read_prop_ledger(db, game_id="g1")}
    assert rows["points"]["actual_value"] == 25.0
    assert rows["points"]["error"] == pytest.approx(1.5)
    assert rows["points"]["hit"] is True
    assert rows["rebounds"]["hit"] is None, "the other market is still ungraded"


def test_recording_an_outcome_without_a_prediction_does_not_invent_one(tmp_path):
    """An outcome with no snapshot has no pick to grade. Inventing one would
    report a prediction nobody made."""
    db = _db(tmp_path)
    store.insert_player_outcome(db, game_id="g9", player_id="p9", stat="points",
                               actual_value=30.0, recorded_at=PRE_TIP)
    assert read_prop_ledger(db, game_id="g9") == []


def test_reading_an_unknown_game_returns_empty_not_an_error(tmp_path):
    assert read_prop_ledger(_db(tmp_path), game_id="nope") == []


# --------------------------------------------------------------------------
# The player-ID join
# --------------------------------------------------------------------------

ESPN_PLAYERS = [
    {"player_id": "111", "player_name": "Jayson Tatum", "team": "BOS"},
    {"player_id": "222", "player_name": "Bam Adebayo", "team": "MIA"},
]


def test_join_matches_on_normalised_name_and_team():
    assert join_player_ids([("jayson tatum", "BOS")], ESPN_PLAYERS) == ["111"]


def test_join_never_guesses_a_near_miss():
    """An unmatched prop is logged and skipped, never fuzzy-matched. A wrong
    line on a real player is worse than a missing line."""
    for name, team in (
        ("Zzz Unknown", "XXX"),
        ("Jayson", "BOS"),          # partial name
        ("Jayson Tatum Jr", "BOS"),  # different player
        ("Tatum", "BOS"),             # surname only
    ):
        assert join_player_ids([(name, team)], ESPN_PLAYERS) == [], (
            f"guessed a match for {name!r}"
        )


def test_join_will_not_match_a_player_on_the_wrong_team():
    """Same person, different team, is not a match here — the props feed's team
    is part of the key, and a transfer must not inherit the old row."""
    assert join_player_ids([("Jayson Tatum", "MIA")], ESPN_PLAYERS) == []


def test_join_preserves_input_order_and_drops_only_unmatched(tmp_path, caplog):
    names = [("Jayson Tatum", "BOS"), ("Nobody Here", "XXX"), ("Bam Adebayo", "MIA")]
    assert join_player_ids(names, ESPN_PLAYERS) == ["111", "222"]

def test_a_contradicting_outcome_is_refused_even_without_the_unique_index(tmp_path):
    """Why the ledger uses `insert_player_outcome` and not a plain insert.

    `game_player_outcomes` carries a unique index, but `_ensure_unique_indexes`
    deliberately LEAVES IT OUT on a legacy database whose existing rows conflict,
    because rule 1 forbids deleting recorded facts. An insert-then-catch writer
    therefore succeeds on such a database and quietly files a second, different
    value for a key already recorded -- the MAE then depends on which row a
    reader happens to see.

    `insert_player_outcome` checks first, so it is correct either way. This
    simulates the index-less database directly, by dropping the index.
    """
    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    db = str(db_path)
    with store.get_connection(db) as conn:
        conn.execute("DROP INDEX IF EXISTS uq_game_player_outcomes_key")
        conn.commit()

    store.insert_player_outcome(db, game_id="g1", player_id="p1", stat="points",
                                actual_value=20.0, recorded_at=PRE_TIP)
    # Same key, different value: a contradiction, and it must be loud.
    with pytest.raises(store.ConflictingOutcome):
        store.insert_player_outcome(db, game_id="g1", player_id="p1", stat="points",
                                    actual_value=99.0, recorded_at=PRE_TIP_LATER)

    with store.get_connection(db) as conn:
        rows = conn.execute(
            "SELECT actual_value FROM game_player_outcomes "
            "WHERE game_id='g1' AND player_id='p1' AND stat='points'"
        ).fetchall()
    assert [r[0] for r in rows] == [20.0], (
        "a second, contradicting outcome was filed on a database with no index"
    )
