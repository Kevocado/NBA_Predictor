import pandas as pd
from datetime import date

from nba_predictor.features.context import current_streak, game_flags, is_high_altitude
from nba_predictor.features.four_factors import add_rolling_four_factors
from nba_predictor.features.rest_travel import (
    compute_rest_days,
    congestion_flags,
    fatigue_index,
    is_back_to_back,
    rolling_travel_miles,
    timezone_change_count,
)

#: How many days of travel a team is carrying into a game. Seven is the week,
#: which is what a road trip is: two games in three nights, a flight home, a
#: home stand. A longer window makes this a season total with a week's name.
TRAVEL_WINDOW_DAYS = 7

FEATURE_COLUMNS = [
    "home_efg_pct_roll", "home_tov_rate_roll", "home_orb_pct_roll", "home_ft_rate_roll",
    "away_efg_pct_roll", "away_tov_rate_roll", "away_orb_pct_roll", "away_ft_rate_roll",
    "home_power_rating", "away_power_rating", "power_rating_diff",
    "home_rest_days", "away_rest_days", "home_back_to_back", "away_back_to_back",
    "home_travel_miles", "away_travel_miles",
    "home_timezone_changes", "away_timezone_changes",
    "home_three_in_four", "away_three_in_four",
    "home_four_in_six", "away_four_in_six",
    "home_fatigue_index", "away_fatigue_index",
    "home_missing_value", "away_missing_value",
    "home_streak", "away_streak",
    "is_high_altitude", "conference_game", "division_game",
]

#: Columns that are absent from the incoming frame and are meant to be, filled
#: with 0.0 so a caller that does not have the data still gets a usable frame.
#:
#: `home_fatigue_index` was in this list AND in FEATURE_COLUMNS, which is how it
#: stayed a constant zero for so long: the zero-fill made the column exist, and
#: nothing ever checked that it moved. It is computed now, so it is not here.
_OPTIONAL_COLUMNS_DEFAULT_ZERO = [
    "home_power_rating", "away_power_rating",
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


def build_feature_frame(
    games: pd.DataFrame, carry_over_weight: float | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Features for `games`.

    `carry_over_weight` is passed to `add_rolling_four_factors`; None (the
    default, and every production caller) is today's behaviour exactly.
    """
    games = games.copy()

    for col in _OPTIONAL_COLUMNS_DEFAULT_ZERO:
        if col not in games.columns:
            games[col] = 0.0
    games["power_rating_diff"] = games["home_power_rating"] - games["away_power_rating"]

    long_form = add_rolling_four_factors(
        _long_format_box_scores(games), carry_over_weight_value=carry_over_weight
    )
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

    #: Per team, in date order: the dates it played and the ABBREVIATION of the
    #: venue it played in. A home game's venue is the team itself; a road game's
    #: is the opponent. `rolling_travel_miles` and `timezone_change_count` read
    #: consecutive pairs of these, so the order matters and must be date order.
    team_game_dates: dict[str, list[str]] = {}
    team_game_venues: dict[str, list[str]] = {}

    home_last_game: dict[str, str] = {}
    away_last_game: dict[str, str] = {}
    home_results: dict[str, list[str]] = {}
    away_results: dict[str, list[str]] = {}
    rest_days_home, rest_days_away = [], []
    streak_home, streak_away = [], []
    travel_home, travel_away = [], []
    tz_home, tz_away = [], []
    congestion_home, congestion_away = [], []
    fatigue_home, fatigue_away = [], []

    for _, row in games.iterrows():
        home, away, game_date = row["home_team"], row["away_team"], row["game_date"]

        rest_days_home.append(compute_rest_days(game_date, home_last_game.get(home)))
        rest_days_away.append(compute_rest_days(game_date, away_last_game.get(away)))
        streak_home.append(current_streak(home_results.get(home, [])))
        streak_away.append(current_streak(away_results.get(away, [])))

        # Travel and congestion are built from each team's OWN games strictly
        # before this one -- the venue list is read before either side is
        # appended, so no game is ever its own history.
        for side, team, rest, miles, tz, congested, fatigue in (
            ("home", home, rest_days_home[-1], travel_home, tz_home, congestion_home, fatigue_home),
            ("away", away, rest_days_away[-1], travel_away, tz_away, congestion_away, fatigue_away),
        ):
            dates = team_game_dates.get(team, [])
            venues = team_game_venues.get(team, [])
            if len(dates) >= 2:
                window = [
                    venues[i] for i in range(len(dates) - 1)
                    if date.fromisoformat(dates[i]).toordinal()
                    >= date.fromisoformat(game_date).toordinal() - (TRAVEL_WINDOW_DAYS - 1)
                ]
                # The last game before this one is where the team currently is,
                # so the trip to its current venue is the one that counts.
                window.append(venues[-1])
                miles.append(rolling_travel_miles(window))
                tz.append(timezone_change_count(window))
                flags = congestion_flags(dates, game_date)
                congested.append(flags)
                fatigue.append(
                    fatigue_index(miles[-1], tz[-1], rest)
                )
            else:
                miles.append(0.0)
                tz.append(0)
                congested.append({"three_in_four": False, "four_in_six": False})
                fatigue.append(fatigue_index(0.0, 0, rest))

        home_last_game[home] = game_date
        away_last_game[away] = game_date
        if pd.notna(row["home_win"]):
            home_results.setdefault(home, []).append("W" if row["home_win"] == 1 else "L")
            away_results.setdefault(away, []).append("L" if row["home_win"] == 1 else "W")

        # The venue of a game is the arena it is played in, which is the HOME
        # team's own. Recording the team instead would make every team's location
        # history a list of its own abbreviation, and haversine(X, X) is zero --
        # travel would compute as always 0 and look like a real measurement.
        team_game_dates.setdefault(home, []).append(game_date)
        team_game_dates.setdefault(away, []).append(game_date)
        team_game_venues.setdefault(home, []).append(home)
        team_game_venues.setdefault(away, []).append(home)

    games["home_rest_days"] = rest_days_home
    games["away_rest_days"] = rest_days_away
    games["home_back_to_back"] = [is_back_to_back(d) for d in rest_days_home]
    games["away_back_to_back"] = [is_back_to_back(d) for d in rest_days_away]
    games["home_streak"] = streak_home
    games["away_streak"] = streak_away
    games["home_travel_miles"] = travel_home
    games["away_travel_miles"] = travel_away
    games["home_timezone_changes"] = tz_home
    games["away_timezone_changes"] = tz_away
    games["home_fatigue_index"] = fatigue_home
    games["away_fatigue_index"] = fatigue_away
    games["home_three_in_four"] = [f["three_in_four"] for f in congestion_home]
    games["away_three_in_four"] = [f["three_in_four"] for f in congestion_away]
    games["home_four_in_six"] = [f["four_in_six"] for f in congestion_home]
    games["away_four_in_six"] = [f["four_in_six"] for f in congestion_away]

    games["is_high_altitude"] = games["home_team"].apply(is_high_altitude)
    flags = games.apply(lambda r: game_flags(r["home_team"], r["away_team"]), axis=1, result_type="expand")
    games["conference_game"] = flags["conference_game"]
    games["division_game"] = flags["division_game"]

    games = games.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return games, FEATURE_COLUMNS


def build_training_frame(
    games: pd.DataFrame, carry_over_weight: float | None = None
) -> tuple[pd.DataFrame, list[str]]:
    return build_feature_frame(games, carry_over_weight=carry_over_weight)
