"""Lagged availability features: causal, unplayed-safe, numerically the evaluated reference."""
from collections import defaultdict

import numpy as np
import pandas as pd

from nba_predictor.features.availability import AVAILABILITY_COLUMNS, compute_availability_features
from nba_predictor.features.build import FEATURE_COLUMNS, build_feature_frame

TEAMS = ["BOS", "NYK", "LAL", "DEN"]


def _world(n_games=40, seed=3):
    rng = np.random.default_rng(seed)
    games, rows = [], []
    for i in range(n_games):
        h, a = [str(t) for t in rng.choice(TEAMS, 2, replace=False)]
        gid, day = f"g{i:03d}", f"2025-11-{1 + i // 2:02d}"
        games.append(dict(game_id=gid, game_date=day, home_team=h, away_team=a))
        for team in (h, a):
            for k in range(10):
                pid = f"{team}{k}"
                m = 0.0 if rng.random() < 0.2 else float(rng.integers(5, 38))
                rows.append(dict(game_id=gid, game_date=day, team=team, player_id=pid, minutes=m,
                                 points=m / 3, rebounds=m / 8, assists=m / 10))
    return pd.DataFrame(games), pd.DataFrame(rows)


def _reference(P, games):
    """Verbatim logic of the evaluated predictor-hub feats.availability_proxy (NaN -> 0.0)."""
    by = {k: g for k, g in P.groupby(["game_id", "team"])}
    order = defaultdict(list)
    for r in games.sort_values(["game_date", "game_id"]).itertuples():
        order[r.home_team].append(r.game_id)
        order[r.away_team].append(r.game_id)
    feat = {}
    for team, gids in order.items():
        cm, cp, prev = defaultdict(float), defaultdict(float), None
        for gid in gids:
            g = by.get((gid, team))
            if prev is None or not cm:
                ms = ps = 0.0
            else:
                top = sorted(cm, key=cm.get, reverse=True)[:8]
                absent = [p for p in top if p not in prev]
                ms = sum(cm[p] for p in absent) / sum(cm[p] for p in top)
                ps = sum(cp[p] for p in absent) / sum(cp[p] for p in top)
            feat[(gid, team)] = (ms, ps)
            if g is not None:
                prev = set(g.loc[g.minutes > 0, "player_id"])
                for r in g.itertuples():
                    cm[r.player_id] += r.minutes
                    cp[r.player_id] += r.points + 1.2 * r.rebounds + 1.5 * r.assists
    return pd.DataFrame({
        "game_id": games.game_id,
        "home_absent_min_share": [feat[(g, t)][0] for g, t in zip(games.game_id, games.home_team)],
        "away_absent_min_share": [feat[(g, t)][0] for g, t in zip(games.game_id, games.away_team)],
        "home_absent_prod_share": [feat[(g, t)][1] for g, t in zip(games.game_id, games.home_team)],
        "away_absent_prod_share": [feat[(g, t)][1] for g, t in zip(games.game_id, games.away_team)],
    })


def test_matches_reference_exactly():
    games, P = _world()
    got = compute_availability_features(games, P)
    want = _reference(P, games)
    assert list(got.columns) == ["game_id", *AVAILABILITY_COLUMNS]
    assert (got[AVAILABILITY_COLUMNS].to_numpy() - want[AVAILABILITY_COLUMNS].to_numpy()).__abs__().max() == 0
    assert got[AVAILABILITY_COLUMNS].to_numpy().max() > 0  # the test is not vacuous


def test_a_games_own_box_score_never_moves_its_features():
    games, P = _world()
    base = compute_availability_features(games, P).set_index("game_id")
    target = "g020"
    P2 = P.copy()
    m = P2.game_id == target
    P2.loc[m, "minutes"] = 99.0
    P2.loc[m, "points"] = 500.0
    P2 = P2[~(m & P2.player_id.str.endswith(("0", "1")))]  # and drop two players from it
    alt = compute_availability_features(games, P2).set_index("game_id")
    pd.testing.assert_series_equal(base.loc[target], alt.loc[target])
    before = games.game_id < target
    pd.testing.assert_frame_equal(base[before.values], alt[before.values])


def test_no_player_history_is_the_documented_zero_default():
    games, P = _world(10)
    for players in (None, P.iloc[0:0]):
        out = compute_availability_features(games, players)
        assert (out[AVAILABILITY_COLUMNS] == 0.0).all().all()
    first = compute_availability_features(games, P).set_index("game_id").loc["g000"]
    assert (first == 0.0).all()  # a team's first game has no history


def test_unplayed_games_use_last_completed_game_and_do_not_pollute_state():
    games, P = _world(30)
    last = games.iloc[-1]
    upcoming = pd.DataFrame([
        dict(game_id="u1", game_date="2025-12-30", home_team=last.home_team, away_team=last.away_team),
        dict(game_id="u2", game_date="2025-12-31", home_team=last.home_team, away_team=last.away_team),
    ])
    out = compute_availability_features(pd.concat([games, upcoming], ignore_index=True), P).set_index("game_id")
    assert not out.isna().any().any()
    # the first upcoming game is computed from the last completed game; the second sees the same state
    assert (out.loc["u1"] == out.loc["u2"]).all()
    # and adding upcoming games leaves every played game's features untouched
    played = compute_availability_features(games, P).set_index("game_id")
    pd.testing.assert_frame_equal(played, out.loc[played.index])


def test_build_feature_frame_carries_the_columns_for_upcoming_games():
    assert set(AVAILABILITY_COLUMNS) <= set(FEATURE_COLUMNS)
    games, P = _world(60)
    rng = np.random.default_rng(1)
    box = {f"{s}_{f}": rng.integers(20, 90, len(games)) for s in ("home", "away")
           for f in ("fgm", "fga", "fg3m", "tov", "oreb", "dreb", "fta", "pts")}
    full = games.assign(**box)
    full["home_win"] = (full.home_pts > full.away_pts).astype(float)
    up = pd.DataFrame([dict(game_id="up", game_date="2026-01-15", home_team="BOS", away_team="NYK")])
    frame, cols = build_feature_frame(pd.concat([full, up], ignore_index=True), players=P)
    row = frame[frame.game_id == "up"]
    assert len(row) == 1 and not row[cols].isna().any().any()
    assert (frame[AVAILABILITY_COLUMNS] > 0).any().any()


# --- the rotation is ranked within a production-sized window, not over all history --------------------------

def _three_seasons():
    """BOS has a star ("X") only in season 2022; from 2023 on the same new core plays. Two teams, 12 games a season."""
    games, rows = [], []
    gid = 0
    for season, start in ((2022, "2022-11-"), (2023, "2023-11-"), (2024, "2024-11-")):
        for i in range(12):
            day = f"{start}{1 + i:02d}"
            games.append(dict(game_id=f"g{gid:03d}", game_date=day, home_team="BOS", away_team="NYK"))
            for team in ("BOS", "NYK"):
                core = ["X", "a", "b", "c", "d", "e", "f", "g"] if (team == "BOS" and season == 2022) else [f"{team}{k}" for k in range(8)]
                for pid in core:
                    m = 40.0 if pid == "X" else 30.0
                    rows.append(dict(game_id=f"g{gid:03d}", game_date=day, team=team, player_id=pid, minutes=m, points=m / 3, rebounds=m / 8, assists=m / 10))
            gid += 1
    return pd.DataFrame(games), pd.DataFrame(rows)


def test_by_season_equals_the_plain_function_on_a_single_season_frame():
    from nba_predictor.features.availability import compute_availability_by_season

    games, players = _world()
    pd.testing.assert_frame_equal(compute_availability_by_season(games, players), compute_availability_features(games, players))


def test_a_departed_players_old_minutes_do_not_make_the_new_core_look_absent():
    from nba_predictor.features.availability import compute_availability_by_season

    games, players = _three_seasons()
    plain = compute_availability_features(games, players).set_index("game_id")
    by_season = compute_availability_by_season(games, players).set_index("game_id")
    opener_2024 = games[games.game_date >= "2024-11-01"].game_id.iloc[1]    # 2nd game: has a previous game to compare
    # Over all history the departed 2022 star still tops BOS's cumulative minutes, so he counts as "absent last game".
    assert plain.loc[opener_2024, "home_absent_min_share"] > 0.05
    # Ranked from the previous season on (2023, 2024), the rotation is the people actually playing.
    assert by_season.loc[opener_2024, "home_absent_min_share"] == 0.0
    assert list(by_season.index) == list(games.game_id)


def test_build_feature_frame_uses_the_season_scoped_availability():
    games, players = _three_seasons()
    raw = games.assign(home_pts=100, away_pts=95, home_win=1, home_fgm=40, home_fga=88, home_fg3m=12, home_tov=11, home_oreb=9, home_dreb=32, home_fta=20,
                       away_fgm=38, away_fga=90, away_fg3m=10, away_tov=13, away_oreb=10, away_dreb=30, away_fta=18)
    frame, _ = build_feature_frame(raw, players=players)
    row = frame[frame.game_id == games[games.game_date >= "2024-11-01"].game_id.iloc[1]].iloc[0]
    assert row["home_absent_min_share"] == 0.0
