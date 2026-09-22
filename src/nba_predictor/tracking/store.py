import sqlite3
from contextlib import contextmanager
from pathlib import Path

# The tracking DB can live on network-attached storage (Azure Files, for
# persistence across container restarts), where a lock briefly held by a
# just-restarted process takes longer to clear than sqlite3's 5s default
# timeout. 30s gives that lock time to release instead of failing fast
# with "database is locked".
_CONNECT_TIMEOUT_SECONDS = 30

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


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path, timeout=_CONNECT_TIMEOUT_SECONDS) as conn:
        conn.executescript(SCHEMA)
        _ensure_point_column(conn)


def _ensure_point_column(conn: sqlite3.Connection) -> None:
    """Adds `point` to a game_market_predictions table created before this
    column existed. CREATE TABLE IF NOT EXISTS above won't add it to an
    already-existing table, so this migration covers any DB file left over
    from a prior deploy."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(game_market_predictions)")}
    if "point" not in columns:
        conn.execute("ALTER TABLE game_market_predictions ADD COLUMN point REAL")
        conn.commit()


@contextmanager
def get_connection(db_path: Path):
    conn = sqlite3.connect(db_path, timeout=_CONNECT_TIMEOUT_SECONDS)
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
    with get_connection(db_path) as conn:
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
            "SELECT * FROM game_player_outcomes WHERE game_id = ?",
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
