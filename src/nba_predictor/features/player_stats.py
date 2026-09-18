import pandas as pd

PLAYER_STATS = ["points", "rebounds", "assists", "fg3m", "minutes"]
PLAYER_FEATURE_COLUMNS = [f"{stat}_roll" for stat in PLAYER_STATS]


def build_player_feature_frame(player_games: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Causal per-player rolling features: for each (player_id, game_date)
    row, a shift(1).rolling(window=10).mean() of points/rebounds/assists/
    fg3m/minutes over that player's real prior games only — same pattern
    as features/four_factors.py::add_rolling_four_factors, applied per
    player instead of per team. A row with no stat value yet (an upcoming
    game) never enters its own rolling average (shift(1) excludes it) or
    any other row's, since pandas rolling means skip NaN.
    """
    games = player_games.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    for stat in PLAYER_STATS:
        games[f"{stat}_roll"] = games.groupby("player_id")[stat].transform(
            lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
        )
    games = games.dropna(subset=PLAYER_FEATURE_COLUMNS).reset_index(drop=True)
    return games, PLAYER_FEATURE_COLUMNS
