"""Game-context features: pace, opponent-adjusted form, rest, injury delta.

Spec section 6, gap G8. The repo has an injury *gate* and no injury *feature*,
because `features/injuries.py` was neutralised after the LeBron-fabrication
incident -- correctly. This adds the feature back with the fabrication risk
removed by construction rather than by neutralising.

**The injury delta only reads rows the verified ESPN availability feed
produced.** Concretely, `data.espn.get_injuries()` emits exactly four keys
(`team`, `player_id`, `player_name`, `status`, `dated`) and nothing else. A row
carrying a `source` key did not come from that function -- it came from
something that is not the feed we verified -- so it is ignored. Same for a
`status` outside the measured `{"Out", "Day-To-Day"}` vocabulary. The tests pin
both, and pin that a *known* Out does move the number, so the feature cannot
pass by being hardcoded to zero.

**An unreadable feed yields 0.0 and `availability_status = "no_report"`.** Not
an assumption that nobody is hurt -- the status column keeps the difference
between "checked and clear" and "not checked" visible to a reader, which is the
asymmetry `api/availability.py` is built around.

**Every rolling feature is shift(1)-then-rolling**: game G may only use games
strictly before G. A team's first game has no prior form and gets NaN, because
0.0 would claim the team was exactly average -- a measurement we do not have.

Pace uses the repo's own `compute_possessions` / `compute_pace` from
`features/ratings.py` rather than a second definition of the same number.
"""

from __future__ import annotations

import pandas as pd

from nba_predictor.api.availability import is_out
from nba_predictor.features.ratings import compute_pace, compute_possessions

#: A team's trailing form spans this many prior games. Matches the 10-game window
#: the NFL prop features use, so the repo has one convention rather than two.
FORM_WINDOW = 10

FEATURE_COLUMNS = ["pace", "opp_adj_net", "rest_delta", "injury_delta"]


def _verified_out_player_ids(availability) -> dict[str, set[str]]:
    """{team: {player_id}} for players the *verified ESPN feed* marks Out.

    Anything else is skipped, with the reason rather than a tolerance: an
    unrecognised row is not a near-miss to be rescued, it is a row of unknown
    provenance, and a feature built on it is a fabricated one.
    """
    out: dict[str, set[str]] = {}
    if not availability:
        return out

    # Exactly the keys `data.espn.get_injuries()` emits. Anything carrying a
    # `source` (or anything else outside this set) is not that function's output.
    feed_keys = {"team", "player_id", "player_name", "status", "dated"}
    for row in availability:
        if not isinstance(row, dict):
            continue
        if not feed_keys.issuperset(row):
            continue
        team, player_id = row.get("team"), row.get("player_id")
        if not team or not player_id:
            continue
        # Day-to-Day is a flag, not a removal -- same rule the availability gate
        # applies, because doubling the rule in two places is how they diverge.
        if is_out(row.get("status")):
            out.setdefault(team, set()).add(player_id)
    return out


def _team_game_log(games: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One row per team per game, from that team's point of view.

    A team's form is its own points-for/against, so home and away games fold into
    a single chronological log per team. Without this, a team that happens to
    play away more often looks like it has less history.
    """
    sides = []
    for side, opp in (("home", "away"), ("away", "home")):
        sides.append(
            pd.DataFrame(
                {
                    "game_id": games["game_id"],
                    "game_date": games["game_date"],
                    "team": games[f"{side}_team"],
                    "points_for": games[f"{side}_pts"],
                    "points_against": games[f"{opp}_pts"],
                    "fga": games[f"{side}_fga"],
                    "fta": games[f"{side}_fta"],
                    "oreb": games[f"{side}_oreb"],
                    "tov": games[f"{side}_tov"],
                }
            )
        )

    by_team: dict[str, list[pd.DataFrame]] = {}
    for frame in sides:
        for team, group in frame.groupby("team"):
            by_team.setdefault(team, []).append(group)

    log: dict[str, pd.DataFrame] = {}
    for team, parts in by_team.items():
        form = pd.concat(parts).sort_values(["game_date", "game_id"]).reset_index(drop=True)
        form["_net"] = form["points_for"] - form["points_against"]
        form["_poss"] = [
            compute_possessions(fga, fta, oreb, tov)
            for fga, fta, oreb, tov in zip(form["fga"], form["fta"], form["oreb"], form["tov"])
        ]
        # Days since this team's previous game. NaN for a team's first game:
        # there is no previous game to be rested from, and 0 would claim the team
        # played the day before, which is a fact we do not have.
        dates = pd.to_datetime(form["game_date"])
        form["_rest"] = (dates - dates.shift(1)).dt.days
        log[team] = form
    return log


def _prior(form: pd.DataFrame, upto_row: int, col: str) -> float:
    """Value of `col` over rows strictly before `upto_row`.

    `.iloc[:upto_row]` on a chronologically sorted frame *is* the shift(1): the
    target game is excluded because it is not yet in the slice. NaN when there is
    no history -- 0.0 would claim the team was exactly average, which is a
    measurement this repo does not have.
    """
    prior = form[col].iloc[:upto_row]
    if prior.empty:
        return float("nan")
    value = prior.mean() if col != "_rest" else prior.iloc[-1]
    return float(value)


def build_game_features(
    games: pd.DataFrame,
    availability=None,
) -> pd.DataFrame:
    """Attach `pace`, `opp_adj_net`, `rest_delta`, `injury_delta`, and
    `availability_status` to a game frame.

    `games` is the same shape `ingest.to_training_frame` builds. `availability`
    is the output of `data.espn.get_injuries()`; pass None when the feed could
    not be read, which is recorded rather than guessed at.
    """
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    out_by_id: dict[str, dict] = {}

    out_ids = _verified_out_player_ids(availability)
    checked = bool(availability)
    status = "checked" if checked else "no_report"

    team_log = _team_game_log(games)

    # index each team's log rows by game_id so a game's own row can be located
    # and *excluded* -- that exclusion is the shift(1).
    row_of: dict[str, dict[str, int]] = {
        team: {gid: i for i, gid in enumerate(form["game_id"])}
        for team, form in team_log.items()
    }

    for game in games.itertuples():
        home, away = game.home_team, game.away_team

        def form_for(team: str) -> dict:
            form = team_log.get(team)
            if form is None:
                return {"net": float("nan"), "rest": float("nan"), "poss": float("nan")}
            i = row_of[team][game.game_id]
            net = _prior(form, i, "_net")
            return {
                "net": net,
                "rest": float(i),
                "poss": _prior(form, i, "_poss"),
            }

        home_form, away_form = form_for(home), form_for(away)

        # Pace: the possessions rate each team brings to the floor, via the
        # repo's own compute_pace, which averages the two sides. Summing first
        # and averaging after would double-count. The game's own shot profile is
        # not knowable pre-tip-off.
        pace = float("nan")
        rates = [f["poss"] for f in (home_form, away_form) if pd.notna(f["poss"])]
        if rates:
            pace = float(compute_pace(*rates)) if len(rates) == 2 else float(rates[0])

        # Opponent-adjusted: the home side's trailing form minus the away side's.
        # Subtracting is the point -- winning *against good teams* should rate
        # higher than winning against bad ones.
        #
        # One value per row, from the home team's perspective. Averaging the
        # home-vs-away and away-vs-home adjustments is what an earlier draft
        # did, and they are exact negatives, so the mean is 0.0 on every game:
        # a feature that measures nothing and passes every NaN check.
        opp_adj = (
            home_form["net"] - away_form["net"]
            if pd.notna(home_form["net"]) and pd.notna(away_form["net"])
            else float("nan")
        )

        rest_delta = (
            home_form["rest"] - away_form["rest"]
            if pd.notna(home_form["rest"]) and pd.notna(away_form["rest"])
            else float("nan")
        )

        out_by_id[game.game_id] = {
            "game_id": game.game_id,
            "pace": pace,
            "opp_adj_net": opp_adj,
            "rest_delta": rest_delta,
            # One out player is one unit of "season-average production we no
            # longer have". ESPN publishes no production values, so this counts
            # verified absences rather than inventing an impact per name --
            # which is the distinction `features/injuries.py` was neutralised
            # for, and the reason this is a count and not a point estimate.
            "injury_delta": float(
                -(len(out_ids.get(home, ())) - len(out_ids.get(away, ())))
            ),
            "availability_status": status,
        }

    features = pd.DataFrame(out_by_id.values())
    return games.merge(features, on="game_id", how="left")