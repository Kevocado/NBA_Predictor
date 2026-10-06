"""Game-context features: pace, opponent-adjusted rating, rest, injury delta.

Spec section 6 / gap G8. The repo has an injury *gate* (Out removes, Day-to-Day
flags) and no injury *feature*, because `features/injuries.py` was neutralised
after the LeBron-fabrication incident and correctly so.

Three properties are pinned, in order of how badly they bite.

1. **No fabrication, ever.** An injury delta may only come from rows the
   verified ESPN availability feed produced. A row from any other source, or a
   status outside the feed's measured vocabulary, must not move the number by
   one float. A delta that moves on invented data is a fabricated feature with a
   real number attached, which is the exact thing that burned this repo before.

2. **shift(1)-then-rolling.** A game-G feature may only use games strictly
   before G. A rolling feature that includes its own target game silently
   inflates every score downstream.

3. **A feature column that is all-NaN is a broken pull, not "no signal"**
   (spec section 12.2). Coverage is asserted per column so a broken feed fails
   here rather than reading as a quiet absence of signal a season later.
"""

import numpy as np
import pandas as pd
import pytest

from nba_predictor.features.game_context import build_game_features

FEATURES = ["pace", "opp_adj_net", "rest_delta", "injury_delta"]


def make_games(n_days: int = 60, n_teams: int = 6) -> pd.DataFrame:
    """`n_days` of games, several a night, across `n_teams` -- the real shape."""
    rng = np.random.default_rng(0)
    teams = ["BOS", "MIA", "LAL", "GSW", "DEN", "PHX"][:n_teams]
    dates = pd.date_range("2025-10-22", periods=n_days).astype(str)
    strength = dict(zip(teams, rng.normal(0, 6, n_teams)))

    rows = []
    for d, day in enumerate(dates):
        for k in range(n_teams // 2):
            home, away = teams[2 * k], teams[2 * k + 1]
            margin = float(strength[home] - strength[away] + rng.normal(0, 8))
            total = float(220 + rng.normal(0, 10))
            rows.append({
                "game_id": f"g{d}-{k}",
                "game_date": day,
                "home_team": home,
                "away_team": away,
                "home_pts": round((total + margin) / 2),
                "away_pts": round((total - margin) / 2),
                "home_fgm": 40, "home_fga": 88, "home_fg3m": 12,
                "home_tov": 11, "home_oreb": 9, "home_dreb": 32, "home_fta": 20,
                "away_fgm": 38, "away_fga": 90, "away_fg3m": 10,
                "away_tov": 13, "away_oreb": 10, "away_dreb": 30, "away_fta": 18,
            })
    return pd.DataFrame(rows)


def espn_injury(team: str, player_id: str, status: str = "Out") -> dict:
    """A row shaped exactly like `data.espn.get_injuries()` emits."""
    return {
        "team": team,
        "player_id": player_id,
        "player_name": f"Player {player_id}",
        "status": status,
        "dated": "2026-10-01T00:00Z",
    }


# --------------------------------------------------------------------------
# 1. No fabrication
# --------------------------------------------------------------------------


def test_injury_delta_moves_only_for_verified_espn_rows():
    """The LeBron incident, as a test. The delta is the production value of
    players the *verified ESPN feed* marks Out. A row from anywhere else must not
    shift it."""
    games = make_games()
    verified = [espn_injury("BOS", "p1")]

    baseline = build_game_features(games, availability=verified)

    for source in ("balldontlie", "nba_api", "sportsbook", "odds_api", "user", ""):
        poisoned = verified + [{
            **espn_injury("BOS", "p2"),
            "source": source,
        }]
        after = build_game_features(games, availability=poisoned)
        assert after["injury_delta"].equals(baseline["injury_delta"]), (
            f"a row sourced from {source!r} moved injury_delta"
        )


def test_injury_delta_ignores_statuses_outside_the_measured_vocabulary():
    """Only "Out" removes; "Day-To-Day" is flagged, not removed. Widening the
    match on a guess is how a note lands on a player the report never flagged
    (`api/availability.py` measures this vocabulary deliberately)."""
    games = make_games()
    baseline = build_game_features(games, availability=[])

    doubtful = build_game_features(
        games, availability=[espn_injury("BOS", "p1", status="Day-To-Day")]
    )
    assert doubtful["injury_delta"].equals(baseline["injury_delta"]), (
        "Day-To-Day removed a player; it is a flag, not a removal"
    )

    for status in ("Out (questionable)", "Probably Out", "doubtful", "out for rest", ""):
        odd = build_game_features(games, availability=[espn_injury("BOS", "p1", status=status)])
        assert odd["injury_delta"].equals(baseline["injury_delta"]), (
            f"status {status!r} moved injury_delta"
        )


def test_injury_delta_is_zero_when_the_feed_is_empty_or_unreadable():
    """An unreadable feed is not a report that came back clean, and neither is a
    claim that nobody is hurt. Zero, with the absence visible."""
    games = make_games()
    empty = build_game_features(games, availability=[])
    unavailable = build_game_features(games, availability=None)

    assert (empty["injury_delta"] == 0.0).all()
    assert (unavailable["injury_delta"] == 0.0).all()
    assert "availability_status" in empty.columns
    assert set(empty["availability_status"]) == {"no_report"}


def test_a_known_out_player_does_move_the_delta():
    """The converse pin: the feature must be able to move at all. A test that
    only asserts 'nothing changes' passes against a feature hardcoded to 0."""
    games = make_games()
    baseline = build_game_features(games, availability=[])
    moved = build_game_features(games, availability=[espn_injury("BOS", "p1")])
    assert not moved["injury_delta"].equals(baseline["injury_delta"]), (
        "injury_delta never moves -- it is not reading the feed"
    )


def test_availability_source_is_reported_on_every_row():
    """A reader must be able to tell 'checked and clear' from 'not checked'."""
    games = make_games()
    with_feed = build_game_features(games, availability=[espn_injury("BOS", "p1")])
    assert set(with_feed["availability_status"]) <= {"no_report", "checked"}


# --------------------------------------------------------------------------
# 2. shift(1)-then-rolling
# --------------------------------------------------------------------------


def test_features_never_use_the_game_they_describe():
    """Game G's features may only use games strictly before G.

    Recomputed independently from the raw frame: for a given game, take each
    side's trailing net margin over ALL of that team's earlier games, and check
    the feature equals `mine - theirs`. If the window included the target game,
    this is the assertion that catches it.
    """
    games = make_games().sort_values(["game_date", "game_id"]).reset_index(drop=True)
    feats = build_game_features(games, availability=[])
    merged = feats.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    checked = 0
    for game in games.itertuples():
        def prior_net(team: str) -> float | None:
            earlier = games[games["game_date"] < game.game_date]
            home = earlier["home_team"] == team
            away = earlier["away_team"] == team
            if not home.any() and not away.any():
                return None
            # .tolist() on each side first: adding two Series aligns on index,
            # which silently produces NaN when the home and away subsets do not
            # share every label.
            home_margins = (
                earlier.loc[home, "home_pts"] - earlier.loc[home, "away_pts"]
            ).tolist()
            away_margins = (
                earlier.loc[away, "away_pts"] - earlier.loc[away, "home_pts"]
            ).tolist()
            margins = home_margins + away_margins
            return float(sum(margins) / len(margins))

        home_net = prior_net(game.home_team)
        away_net = prior_net(game.away_team)
        if home_net is None or away_net is None:
            continue

        row = merged[merged["game_id"] == game.game_id].iloc[0]
        assert row["opp_adj_net"] == pytest.approx(home_net - away_net, abs=1e-6), (
            f"{game.game_id}: opp_adj_net is not (home prior net - away prior net); "
            "the window is including the target game or a later one"
        )
        checked += 1

    assert checked > 20, f"only {checked} games had two-sided history; too weak a check"

def _rotating_games(n_days: int = 14) -> pd.DataFrame:
    """Four teams, pairings rotate nightly, so a team really does miss nights.

    `make_games` pairs the same teams every night, so deleting one night puts
    *both* its teams on a break and the rest difference is always zero. That is
    why the first version of this test could not tell a days-off feature from a
    games-played count -- the fixture could not express the case at all.
    """
    teams = ["AAA", "BBB", "CCC", "DDD"]
    rows = []
    for d, day in enumerate(pd.date_range("2025-11-01", periods=n_days).astype(str)):
        pair = 0 if d % 2 == 0 else 1
        for k in range(2):
            home = teams[(2 * k + pair) % 4]
            away = teams[(2 * k + pair + 1) % 4]
            rows.append({
                "game_id": f"g{d}-{k}", "game_date": day,
                "home_team": home, "away_team": away,
                "home_pts": 110, "away_pts": 105,
                "home_fgm": 40, "home_fga": 88, "home_fg3m": 12,
                "home_tov": 11, "home_oreb": 9, "home_dreb": 32, "home_fta": 20,
                "away_fgm": 38, "away_fga": 90, "away_fg3m": 10,
                "away_tov": 13, "away_oreb": 10, "away_dreb": 30, "away_fta": 18,
            })
    return pd.DataFrame(rows)


def test_rest_delta_reflects_days_off_not_games_played():
    """`rest_delta` is home's days-since-last-game minus the away team's.

    An earlier version returned the row index of the game in the team's own log
    -- a games-played count wearing the name "rest days".
    """
    games = _rotating_games()
    skipped = "2025-11-05"
    games = games[
        ~((games["game_date"] == skipped)
          & (((games["home_team"] == "AAA") & (games["away_team"] == "BBB"))
            | ((games["home_team"] == "BBB") & (games["away_team"] == "AAA"))))
    ].reset_index(drop=True)

    feats = build_game_features(games, availability=[])

    # The night after, the side that sat out has rested two days while the other
    # has rested one. A games-played count gets this BACKWARDS, because the team
    # that sat out has played one *fewer* game -- so the sign is the tell.
    after = feats[feats["game_date"] == "2025-11-06"]
    rested = after[
        after["home_team"].isin(["AAA", "BBB"]) & after["away_team"].isin(["CCC", "DDD"])
    ]
    assert not rested.empty, "fixture produced no game on 2025-11-06 to check"
    expected = 1.0 if rested.iloc[0]["home_team"] in ("AAA", "BBB") else -1.0
    assert float(rested.iloc[0]["rest_delta"]) == pytest.approx(expected), (
        "rest_delta is not days-off; a games-played count gives the opposite sign"
    )

    # Every other night both sides have exactly one day of rest.
    same = feats[feats["game_date"] != "2025-11-06"]["rest_delta"].dropna()
    assert (same == 0).all()


def test_rest_delta_is_home_rest_minus_away_rest_and_is_zero_on_a_back_to_back():
    games = make_games()
    feats = build_game_features(games, availability=[])

    # NaN for a team's first game: no previous game to be rested from.
    assert set(feats["rest_delta"].dropna().unique()) <= {0.0, 1.0, 2.0, 3.0}
    # Two teams play the same night, so their rest days must match.
    same_night = feats.dropna(subset=["rest_delta"]).groupby("game_date")["rest_delta"].nunique()
    assert (same_night == 1).all(), "teams on the same night got different rest counts"


def test_early_games_have_no_history_rather_than_invented_history():
    """A team's first game has no prior form. NaN says so; 0.0 would claim the
    team was exactly average, which is a measurement this repo does not have."""
    games = make_games()
    feats = build_game_features(games, availability=[])
    first_game = feats.groupby("home_team")["opp_adj_net"].apply(lambda s: s.isna().iloc[0])
    assert first_game.all(), "a team's first game was given invented form"


# --------------------------------------------------------------------------
# 3. No all-NaN columns (spec section 12.2)
# --------------------------------------------------------------------------


def test_no_feature_column_is_mostly_nan():
    """An all-NaN column is a broken pull, not the absence of signal. Asserted
    per column so a broken feed fails here rather than a season later."""
    feats = build_game_features(make_games(), availability=[])
    for col in FEATURES:
        coverage = feats[col].notna().mean()
        assert coverage > 0.8, (
            f"{col} is {coverage:.0%} non-NaN -- a broken pull reads as no signal "
            f"(spec 12.2)"
        )


    known = feats["pace"].dropna()
    assert len(known) > 0, "no game had two-sided history"
    assert (known > 60).all(), f"pace below 60 possessions/48 is not NBA pace: {known.min()}"
    assert known.std() > 0, "pace is constant -- the feature is not measuring anything"
