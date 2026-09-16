import sqlite3
from contextlib import contextmanager
from pathlib import Path

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
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_connection(db_path: Path):
    conn = sqlite3.connect(db_path)
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
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO game_market_predictions
                (game_id, market, selection, model_probability, market_probability, edge, bookmaker, american_odds, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (game_id, market, selection, model_probability, market_probability, edge, bookmaker, american_odds, created_at),
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
