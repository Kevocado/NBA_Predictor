"""Player props: the pre-tip pick per player+stat, and the resolved MAE.

The MAE is computed from RESOLVED rows only -- rows with both a prediction and
a recorded actual. It grades the same pick the endpoint serves (the newest
made before tip-off, else the newest, flagged rebuilt), because a "+/-" beside
a projection must describe the number being shown, not some other snapshot of
it.

A backtest row -- a prediction written after the game was played and scored
against that game's result -- contributes nothing. Same reason CFB's track
record has to stop counting post-kickoff picks: judging a look-forward pick
makes the error estimate smaller than the error a real bettor would have seen.
"""

from pathlib import Path

from nba_predictor.tracking import store
from nba_predictor.tracking.timing import latest_by_instant, latest_pre_tip


def _picks_from_rows(rows, game: dict) -> dict[tuple[str, str], dict]:
    """{(player_id, stat): (pick_row, rebuilt)} from one game's snapshot rows."""
    by_key: dict[tuple[str, str], list] = {}
    for row in rows:
        by_key.setdefault((row["player_id"], row["stat"]), []).append(row)

    picks = {}
    for key, key_rows in by_key.items():
        pick = latest_pre_tip(key_rows, game)
        rebuilt = pick is None
        picks[key] = (pick if pick is not None else latest_by_instant(key_rows), rebuilt)
    return picks


def picks_by_player_stat(db_path: Path, game_id: str, game: dict) -> dict[tuple[str, str], dict]:
    """{(player_id, stat): (pick_row, rebuilt)} for one game."""
    return _picks_from_rows(store.get_player_predictions_for_game(db_path, game_id), game)


def actuals_by_player_stat(db_path: Path, game_id: str) -> dict[tuple[str, str], float]:
    return {
        (row["player_id"], row["stat"]): row["actual_value"]
        for row in store.get_player_outcomes_for_game(db_path, game_id)
    }


# SQLite caps the number of bound parameters per statement; stay well under it.
_IN_CHUNK = 500


def resolved_player_props(db_path: Path, schedule: list[dict]) -> list[dict]:
    """Every (stat, prediction, actual) that was actually gradeable.

    Same pick rule as the endpoint, and rebuilt rows are left out: a prediction
    written after the fact is not evidence about how wrong the model was on
    information it did not have.

    Only a game that already has a recorded outcome can contribute, and that is a
    handful of a season's ~1,760 scheduled games. So the outcomes are read once, and
    predictions are read only for those games, in bulk. The per-game version ran two
    queries for EVERY scheduled game (about 3,500) and took ~50 s on the live site.
    """
    games = {g.get("game_id"): g for g in schedule if g.get("game_id")}
    actuals_by_game: dict[str, dict[tuple[str, str], float]] = {}
    rows_by_game: dict[str, list] = {}
    with store.get_connection(db_path) as conn:
        for row in conn.execute("SELECT game_id, player_id, stat, actual_value FROM game_player_outcomes"):
            if row["game_id"] in games:
                actuals_by_game.setdefault(row["game_id"], {})[(row["player_id"], row["stat"])] = row["actual_value"]
        ids = list(actuals_by_game)
        for i in range(0, len(ids), _IN_CHUNK):
            chunk = ids[i : i + _IN_CHUNK]
            marks = ",".join("?" * len(chunk))
            for row in conn.execute(
                f"SELECT * FROM player_prediction_snapshots WHERE game_id IN ({marks}) ORDER BY created_at",
                chunk,
            ):
                rows_by_game.setdefault(row["game_id"], []).append(row)

    resolved = []
    for game in schedule:  # schedule order, exactly as before
        game_id = game.get("game_id")
        if not game_id or game_id not in actuals_by_game:
            continue
        actuals = actuals_by_game[game_id]
        for key, (pick, rebuilt) in _picks_from_rows(rows_by_game.get(game_id, []), game).items():
            if rebuilt or pick is None:
                continue
            actual = actuals.get(key)
            if actual is None:
                continue
            resolved.append(
                {
                    "stat": key[1],
                    "predicted_value": pick["predicted_value"],
                    "actual_value": actual,
                }
            )
    return resolved
