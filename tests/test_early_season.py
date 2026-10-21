import pandas as pd

from nba_predictor.features.early_season import HANDOFF_GAMES, shrunk_projections

COLS = {"threes": "fg3m"}


def _played(last_season_threes, this_season_threes):
    rows = [{"player_id": "p", "game_date": f"2026-04-{i + 1:02d}", "fg3m": v} for i, v in enumerate(last_season_threes)]
    rows += [{"player_id": "p", "game_date": f"2026-10-{20 + i:02d}", "fg3m": v} for i, v in enumerate(this_season_threes)]
    return pd.DataFrame(rows)


def _up(date):
    return pd.DataFrame([{"game_id": "g", "player_id": "p", "game_date": date}])


def test_week_one_is_last_season_not_the_last_few_games():
    out = shrunk_projections(_played([2] * 30 + [5] * 5, []), _up("2026-10-20"), COLS)     # a hot playoff run must not dominate
    assert out[("g", "p", "threes")] == (30 * 2 + 5 * 5) / 35


def test_this_season_takes_over_as_games_accrue():
    out = shrunk_projections(_played([2] * 30, [6, 6, 6]), _up("2026-10-25"), COLS)
    assert out[("g", "p", "threes")] == 0.5 * 6 + 0.5 * 2                                    # n=3, K=3: half and half


def test_after_the_handoff_the_model_keeps_the_number():
    assert shrunk_projections(_played([2] * 30, [3] * HANDOFF_GAMES), _up("2026-11-30"), COLS) == {}


def test_a_player_with_no_last_season_is_left_to_the_model():
    assert shrunk_projections(_played([], [3, 3]), _up("2026-10-25"), COLS) == {}


def test_only_games_before_the_game_are_read():
    out = shrunk_projections(_played([2] * 30, [9]), _up("2026-10-20"), COLS)               # the 20 Oct game itself is not "before"
    assert out[("g", "p", "threes")] == 2.0
