import pandas as pd

from nba_predictor.features.context import current_streak, game_flags, is_high_altitude
from nba_predictor.features.four_factors import add_rolling_four_factors
from nba_predictor.features.rest_travel import compute_rest_days, is_back_to_back

FEATURE_COLUMNS = [
    "home_efg_pct_roll", "home_tov_rate_roll", "home_orb_pct_roll", "home_ft_rate_roll",
    "away_efg_pct_roll", "away_tov_rate_roll", "away_orb_pct_roll", "away_ft_rate_roll",
    "home_power_rating", "away_power_rating", "power_rating_diff",
    "home_rest_days", "away_rest_days", "home_back_to_back", "away_back_to_back",
    "home_fatigue_index", "away_fatigue_index",
    "home_missing_value", "away_missing_value",
    "home_streak", "away_streak",
    "is_high_altitude", "conference_game", "division_game",
]

_OPTIONAL_COLUMNS_DEFAULT_ZERO = [
    "home_power_rating", "away_power_rating",
    "home_fatigue_index", "away_fatigue_index",
    "home_missing_value", "away_missing_value",
]


def _long_format_box_scores(games: pd.DataFrame) -> pd.DataFrame:
    home_rows = games.rename(
        columns={
            "home_team": "team", "home_fgm": "fgm", "home_fga": "fga", "home_fg3m": "fg3m",
            "home_tov": "tov", "home_oreb": "oreb", "away_dreb": "opp_dreb", "home_fta": "fta",
        }
    )[["game_id", "game_date", "team", "fgm", "fga", "fg3m", "tov", "oreb", "opp_dreb", "fta"]]

    away_rows = games.rename(
        columns={
            "away_team": "team", "away_fgm": "fgm", "away_fga": "fga", "away_fg3m": "fg3m",
            "away_tov": "tov", "away_oreb": "oreb", "home_dreb": "opp_dreb", "away_fta": "fta",
        }
    )[["game_id", "game_date", "team", "fgm", "fga", "fg3m", "tov", "oreb", "opp_dreb", "fta"]]

    return pd.concat([home_rows, away_rows], ignore_index=True)


def build_feature_frame(games: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    games = games.copy()

    for col in _OPTIONAL_COLUMNS_DEFAULT_ZERO:
        if col not in games.columns:
            games[col] = 0.0
    games["power_rating_diff"] = games["home_power_rating"] - games["away_power_rating"]

    long_form = add_rolling_four_factors(_long_format_box_scores(games))
    rolled = long_form.set_index(["game_id", "team"])[
        ["efg_pct_roll", "tov_rate_roll", "orb_pct_roll", "ft_rate_roll"]
    ]

    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        merged = games.merge(
            rolled.reset_index(),
            left_on=["game_id", team_col],
            right_on=["game_id", "team"],
            how="left",
        )
        for factor in ["efg_pct_roll", "tov_rate_roll", "orb_pct_roll", "ft_rate_roll"]:
            games[f"{side}_{factor}"] = merged[factor].values

    games = games.sort_values("game_date").reset_index(drop=True)

    home_last_game: dict[str, str] = {}
    away_last_game: dict[str, str] = {}
    home_results: dict[str, list[str]] = {}
    away_results: dict[str, list[str]] = {}
    rest_days_home, rest_days_away = [], []
    streak_home, streak_away = [], []

    for _, row in games.iterrows():
        home, away, game_date = row["home_team"], row["away_team"], row["game_date"]

        rest_days_home.append(compute_rest_days(game_date, home_last_game.get(home)))
        rest_days_away.append(compute_rest_days(game_date, away_last_game.get(away)))
        streak_home.append(current_streak(home_results.get(home, [])))
        streak_away.append(current_streak(away_results.get(away, [])))

        home_last_game[home] = game_date
        away_last_game[away] = game_date
        if pd.notna(row["home_win"]):
            home_results.setdefault(home, []).append("W" if row["home_win"] == 1 else "L")
            away_results.setdefault(away, []).append("L" if row["home_win"] == 1 else "W")

    games["home_rest_days"] = rest_days_home
    games["away_rest_days"] = rest_days_away
    games["home_back_to_back"] = [is_back_to_back(d) for d in rest_days_home]
    games["away_back_to_back"] = [is_back_to_back(d) for d in rest_days_away]
    games["home_streak"] = streak_home
    games["away_streak"] = streak_away

    games["is_high_altitude"] = games["home_team"].apply(is_high_altitude)
    flags = games.apply(lambda r: game_flags(r["home_team"], r["away_team"]), axis=1, result_type="expand")
    games["conference_game"] = flags["conference_game"]
    games["division_game"] = flags["division_game"]

    games = games.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return games, FEATURE_COLUMNS


def build_training_frame(games: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    return build_feature_frame(games)
