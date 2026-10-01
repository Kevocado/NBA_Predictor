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


def picks_by_player_stat(db_path: Path, game_id: str, game: dict) -> dict[tuple[str, str], dict]:
    """{(player_id, stat): (pick_row, rebuilt)} for one game."""
    by_key: dict[tuple[str, str], list] = {}
    for row in store.get_player_predictions_for_game(db_path, game_id):
        by_key.setdefault((row["player_id"], row["stat"]), []).append(row)

    picks = {}
    for key, rows in by_key.items():
        pick = latest_pre_tip(rows, game)
        rebuilt = pick is None
        picks[key] = (pick if pick is not None else latest_by_instant(rows), rebuilt)
    return picks


def actuals_by_player_stat(db_path: Path, game_id: str) -> dict[tuple[str, str], float]:
    return {
        (row["player_id"], row["stat"]): row["actual_value"]
        for row in store.get_player_outcomes_for_game(db_path, game_id)
    }


def resolved_player_props(db_path: Path, schedule: list[dict]) -> list[dict]:
    """Every (stat, prediction, actual) that was actually gradeable.

    Same pick rule as the endpoint, and rebuilt rows are left out: a prediction
    written after the fact is not evidence about how wrong the model was on
    information it did not have.
    """
    resolved = []
    for game in schedule:
        game_id = game.get("game_id")
        if not game_id:
            continue
        actuals = actuals_by_player_stat(db_path, game_id)
        for key, (pick, rebuilt) in picks_by_player_stat(db_path, game_id, game).items():
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
