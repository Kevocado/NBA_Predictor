import sqlite3
import threading
import warnings
from contextlib import contextmanager
from pathlib import Path

_CONNECT_TIMEOUT_SECONDS = 30
# The tracking DB can live on an Azure Files (SMB) mount for persistence
# across container restarts. SQLite's OS-level file locking (fcntl-based)
# isn't honored reliably over that network filesystem -- CREATE TABLE on
# an empty file fails immediately with "database is locked" even with no
# real contention and a generous timeout. Every connection below opens
# with nolock=1 to bypass that broken locking, and _DB_LOCK is what
# actually serializes access instead -- safe only because the container
# app is pinned to exactly one replica, making this process the sole
# writer.
_DB_LOCK = threading.Lock()


def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?nolock=1", uri=True, timeout=_CONNECT_TIMEOUT_SECONDS)

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    model_version TEXT NOT NULL,
    home_win_prob REAL NOT NULL,
    predicted_margin REAL NOT NULL,
    predicted_total REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS game_market_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    market TEXT NOT NULL,
    selection TEXT NOT NULL,
    model_probability REAL NOT NULL,
    market_probability REAL,
    edge REAL,
    bookmaker TEXT,
    american_odds INTEGER,
    point REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS game_forecast_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    snapshot_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS odds_timing_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    market TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    bookmaker TEXT NOT NULL,
    american_odds INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS player_prediction_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    stat TEXT NOT NULL,
    predicted_value REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS game_player_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    stat TEXT NOT NULL,
    actual_value REAL NOT NULL,
    recorded_at TEXT NOT NULL
);
"""

# The one row key this schema ENFORCES, and the only one it can.
#
# Every table above carries `id INTEGER PRIMARY KEY AUTOINCREMENT`. That is a
# PRIMARY KEY, but a surrogate one: it makes each ROW distinct and constrains
# nothing a reader can care about, so two rows can always claim the same
# business key. That was true of every table here until this index.
#
# It is applied to `game_player_outcomes` and ONLY to it, because that is the
# only one of the six whose row key is a FACT rather than a history:
#
#   * `game_player_outcomes` -- one player's actual stat line in one game. There
#     is exactly one truth, a rerun of the ingest cannot produce a second, and
#     the reader grades picks against it. It is the one table where a duplicate
#     silently moves an aggregate (see `player_props._earliest_outcome`, which
#     is the reader half of this same key).
#   * `predictions` and `player_prediction_snapshots` -- deliberately MANY rows
#     per key. Rule 2 of the 2026-10-01 spec (predictor-hub #66) keeps a later
#     rerun as history: it stays in the table and stays in the published
#     per-pick list, and `earliest_recorded` decides which one counts. The
#     scheduled refresh re-presents a rolling window of finished games every
#     day, so duplicates there are the product working, not a bug. A UNIQUE
#     index would not harden them, it would abort the daily ingest and take away
#     the history the rule reads.
#   * `game_market_predictions` -- several rows per (game, market) BY
#     CONSTRUCTION: `refresh_odds` writes both sides of every bookmaker at one
#     instant. Its counted unit is the run (`earliest_run`), not the row.
#   * `game_forecast_snapshots`, `odds_timing_snapshots` -- no writer and no
#     reader anywhere in `src/`, so there is nothing to protect and inventing a
#     key for them would be a claim about a table nothing uses.
#
# The reader key and this index key are the SAME triple, (game_id, player_id,
# stat). They must stay the same: the reader's dedupe is only a statement about
# the rule when the schema enforces the same key, and a reader keyed on a
# column the constraint does not cover is precisely the silent hole this closes
# (`_counting_key` in tracking/timing.py is the reader half).
_UNIQUE_INDEXES = (
    ("uq_game_player_outcomes_key", """
    CREATE UNIQUE INDEX IF NOT EXISTS uq_game_player_outcomes_key
    ON game_player_outcomes (game_id, player_id, stat)
    """),
)


class ConflictingOutcome(Exception):
    """A second, DIFFERENT actual value for a key already recorded.

    Raised instead of writing the row. Recorded outcomes are immutable (rule 1:
    recorded stays recorded) and an outcome is a fact, so a disagreeing value is
    not an update to apply -- it is a contradiction that a human has to see.
    Silently ignoring it (INSERT OR IGNORE) or silently letting scan order
    decide (no constraint, no dedupe on the read) are the two failures this
    module exists to prevent.
    """


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _DB_LOCK, _connect(db_path) as conn:
        conn.executescript(SCHEMA)
        _ensure_point_column(conn)
        _ensure_unique_indexes(conn)


def _ensure_point_column(conn: sqlite3.Connection) -> None:
    """Adds `point` to a game_market_predictions table created before this
    column existed. CREATE TABLE IF NOT EXISTS above won't add it to an
    already-existing table, so this migration covers any DB file left over
    from a prior deploy."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(game_market_predictions)")}
    if "point" not in columns:
        conn.execute("ALTER TABLE game_market_predictions ADD COLUMN point REAL")
        conn.commit()


def _ensure_unique_indexes(conn: sqlite3.Connection) -> None:
    """Adds the enforced row key to any DB file created before it existed.

    SQLite has no `ALTER TABLE ... ADD CONSTRAINT`, so the constraint is a
    unique index, which enforces exactly what a UNIQUE table constraint does:
    a duplicate insert raises `sqlite3.IntegrityError` instead of landing.

    Two properties matter more than the happy path here, because `init_db` runs
    on EVERY boot (`api/app.py`) and in the public-snapshot build:

    * **Idempotent.** `IF NOT EXISTS`, so a second `init_db` is a no-op --
      `test_init_db_is_idempotent` already required that and still passes.
    * **It never takes the service down on a file it cannot fix.** A file
      written before this change can already hold two outcomes for one key, and
      the scheduled refresh re-presents a rolling window of finished games
      every day, so that is the expected shape of an old file rather than a
      corruption. `CREATE UNIQUE INDEX` refuses to build over existing
      conflicts, and rule 1 forbids deleting the rows that block it -- so the
      index is skipped, `conflicting_outcome_keys` still reports them, the
      reader dedupes, and a warning names the count instead of the constraint
      quietly pretending to be there. Silently skipping would be the exact
      degradation this closes: a schema that looks enforced and is not.
    """
    for name, sql in _UNIQUE_INDEXES:
        try:
            conn.execute(sql)
        except sqlite3.IntegrityError:
            conflicts = _conflicting_outcome_keys(conn)
            warnings.warn(
                f"could not build unique index {name}: {len(conflicts)} "
                "(game_id, player_id, stat) key(s) in game_player_outcomes "
                "already hold more than one row, so that index is NOT in "
                "place. Those rows are left alone (recorded stays recorded); "
                "the reader keeps the earliest one per key. "
                "`store.conflicting_outcome_keys` lists them.",
                RuntimeWarning,
                stacklevel=2,
            )
        conn.commit()


def _conflicting_outcome_keys(conn: sqlite3.Connection) -> list[tuple[str, str, str, int]]:
    """`conflicting_outcome_keys` on an already-open connection.

    Split out because `init_db` holds `_DB_LOCK` and `get_connection` takes it
    again -- `threading.Lock` does not re-enter, so the locked public wrapper
    must not be called from inside a locked migration.
    """
    return [
        (row[0], row[1], row[2], row[3])
        for row in conn.execute(
            """
            SELECT game_id, player_id, stat, COUNT(*) AS n_rows
            FROM game_player_outcomes
            GROUP BY game_id, player_id, stat
            HAVING COUNT(*) > 1
            ORDER BY game_id, player_id, stat
            """
        )
    ]


def conflicting_outcome_keys(db_path: Path) -> list[tuple[str, str, str, int]]:
    """`(game_id, player_id, stat, n_rows)` for every duplicated outcome key.

    Ordered so the answer is stable and quotable. Empty on a clean database,
    and empty on every database this version creates.
    """
    with get_connection(db_path) as conn:
        return _conflicting_outcome_keys(conn)


@contextmanager
def get_connection(db_path: Path):
    with _DB_LOCK:
        conn = _connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()


def insert_prediction(
    db_path: Path,
    *,
    game_id: str,
    created_at: str,
    model_version: str,
    home_win_prob: float,
    predicted_margin: float,
    predicted_total: float,
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
                (game_id, created_at, model_version, home_win_prob, predicted_margin, predicted_total)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (game_id, created_at, model_version, home_win_prob, predicted_margin, predicted_total),
        )
        conn.commit()
        return cur.lastrowid


def get_predictions_for_game(db_path: Path, game_id: str) -> list[sqlite3.Row]:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM predictions WHERE game_id = ? ORDER BY created_at",
            (game_id,),
        )
        return cur.fetchall()


def insert_market_prediction(
    db_path: Path,
    *,
    game_id: str,
    market: str,
    selection: str,
    model_probability: float,
    market_probability: float | None,
    edge: float | None,
    bookmaker: str | None,
    american_odds: int | None,
    created_at: str,
    point: float | None = None,
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO game_market_predictions
                (game_id, market, selection, model_probability, market_probability, edge, bookmaker, american_odds, point, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (game_id, market, selection, model_probability, market_probability, edge, bookmaker, american_odds, point, created_at),
        )
        conn.commit()
        return cur.lastrowid


def get_market_predictions_for_game(db_path: Path, game_id: str) -> list[sqlite3.Row]:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM game_market_predictions WHERE game_id = ? ORDER BY created_at",
            (game_id,),
        )
        return cur.fetchall()


def insert_player_prediction(
    db_path: Path,
    *,
    game_id: str,
    player_id: str,
    stat: str,
    predicted_value: float,
    created_at: str,
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO player_prediction_snapshots (game_id, player_id, stat, predicted_value, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (game_id, player_id, stat, predicted_value, created_at),
        )
        conn.commit()
        return cur.lastrowid


def get_player_predictions_for_game(db_path: Path, game_id: str) -> list[sqlite3.Row]:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM player_prediction_snapshots WHERE game_id = ? ORDER BY created_at",
            (game_id,),
        )
        return cur.fetchall()


def get_latest_prediction_for_game(db_path: Path, game_id: str) -> sqlite3.Row | None:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM predictions WHERE game_id = ? ORDER BY created_at DESC LIMIT 1",
            (game_id,),
        )
        return cur.fetchone()


def insert_player_outcome(
    db_path: Path,
    *,
    game_id: str,
    player_id: str,
    stat: str,
    actual_value: float,
    recorded_at: str,
) -> int:
    """Records one player's actual stat line, and returns its row id.

    Idempotent on the same fact, and LOUD on a contradicting one.

    The scheduled refresh (`refresh-data.yml`) runs the ingest over a rolling
    60-day window every day, so this is called again for every recently finished
    game on every run. That is why the check is here rather than left to the
    caller: a plain INSERT would pile a duplicate outcome on top of the one
    already recorded, and nothing downstream could tell which of them the MAE
    was graded against.

    * **Same (game, player, stat), same value** -> the recorded row's id is
      returned and nothing is written. The fact is already on the record;
      writing it twice would add a row that changes no number.
    * **Same key, DIFFERENT value** -> raises `ConflictingOutcome`. Outcomes are
      immutable, so this is not an update to apply; it is a contradiction
      (`store_player_outcomes` counts and logs it rather than letting it abort
      the ingest). `INSERT OR IGNORE` would swallow it and `INSERT OR REPLACE`
      would rewrite a recorded pick -- the two silent outcomes this avoids.
    * **New key** -> inserted.

    Read-then-write inside the connection, not `INSERT ... ON CONFLICT DO
    NOTHING`, and that is deliberate. `get_connection` holds `_DB_LOCK` across
    the whole block, and this module's header records why that lock is enough
    (one replica, so this process is the sole writer), so there is no in-process
    window between the SELECT and the INSERT for the index to close. More to
    the point, SQLite rejects an `ON CONFLICT (game_id, player_id, stat)` target
    that matches no unique index, raising `OperationalError: ON CONFLICT clause
    does not match any PRIMARY KEY or UNIQUE constraint`. That is precisely the
    legacy-file case this module deliberately supports: where existing
    conflicting rows block the index and rule 1 forbids removing them,
    `_ensure_unique_indexes` leaves the index out -- and an upsert would then
    break every write to that file. This form works with or without the index,
    and on a file that has none it is the pre-check that still stops a new
    duplicate.

    The reader is unchanged by any of this and stays load-bearing:
    `player_props._earliest_outcome` still keeps the earliest row per key, so a
    file written before this index existed keeps grading deterministically.
    """
    with get_connection(db_path) as conn:
        existing = conn.execute(
            """
            SELECT id, actual_value FROM game_player_outcomes
            WHERE game_id = ? AND player_id = ? AND stat = ?
            ORDER BY id LIMIT 1
            """,
            (game_id, player_id, stat),
        ).fetchone()
        if existing is not None:
            if float(existing["actual_value"]) != float(actual_value):
                raise ConflictingOutcome(
                    f"game_player_outcomes already records "
                    f"{existing['actual_value']!r} for (game_id={game_id!r}, "
                    f"player_id={player_id!r}, stat={stat!r}) as row "
                    f"{existing['id']}; refusing to record {actual_value!r}. "
                    "Recorded outcomes are immutable."
                )
            return existing["id"]
        cur = conn.execute(
            """
            INSERT INTO game_player_outcomes (game_id, player_id, stat, actual_value, recorded_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (game_id, player_id, stat, actual_value, recorded_at),
        )
        conn.commit()
        return cur.lastrowid


def get_player_outcomes_for_game(db_path: Path, game_id: str) -> list[sqlite3.Row]:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM game_player_outcomes WHERE game_id = ? ORDER BY recorded_at, id",
            (game_id,),
        )
        return cur.fetchall()


def get_all_predictions(db_path: Path) -> list[sqlite3.Row]:
    """The latest prediction per game across the whole tracking DB (for settlement)."""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            SELECT p.* FROM predictions p
            INNER JOIN (
                SELECT game_id, MAX(created_at) AS max_created_at
                FROM predictions
                GROUP BY game_id
            ) latest ON p.game_id = latest.game_id AND p.created_at = latest.max_created_at
            """
        )
        return cur.fetchall()
