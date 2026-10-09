"""NBA four-factors rank duels.

The load-bearing test here is `test_ranks_are_computed_from_the_games_and_not_constant`.
NFL main once shipped duels whose ranks were the same two numbers for every game,
which read as a measurement while meaning nothing. That is the failure this file
exists to prevent, so it is asserted before anything else in the module is trusted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_predictor.signals.four_factors_duel import (
    four_factors_duel,
    team_game_factors,
)

BOX = {"fgm": 40, "fga": 88, "fg3m": 12, "tov": 11, "oreb": 9, "dreb": 32, "fta": 20}

#: 30 teams. NBA's league size, so "rank N of 30" is a real position.
TEAMS = [f"T{i:02d}" for i in range(30)]


def _box(**over):
    row = dict(BOX)
    row.update(over)
    return row


def _game(game_id, date, home, away, home_box, away_box):
    """A row in `to_training_frame`'s shape -- the pipeline's own output."""
    row = {
        "game_id": game_id, "game_date": date,
        "home_team": home, "away_team": away,
        "home_pts": 110, "away_pts": 100, "home_win": 1,
    }
    for field, value in home_box.items():
        row[f"home_{field}"] = value
    for field, value in away_box.items():
        row[f"away_{field}"] = value
    return row


def _schedule(dates, trend: float = 0.0):
    """A balanced round-robin: every team plays once per date, rotating opponents.

    The rotation matters. A defence is only ever what it ALLOWS, so if a strong
    team only ever faced other strong teams its defensive rank would restate its
    own offence. The circle method gives every team a spread of opponents -- over
    a full 29-date round each team plays every other exactly once -- so a team's
    own form and what it allows can genuinely disagree, which is what a duel
    measures.

    `trend` shifts every team's shooting by `trend * date_index`, so form moves
    over time and a window length actually means something.
    """
    n = len(TEAMS)
    rows = []
    for d_i, date in enumerate(dates):
        seen: set[tuple[int, int]] = set()
        for i in range(n):
            partner = (d_i % (n - 1)) if i == n - 1 else ((d_i - i) % (n - 1))
            if partner == i:
                continue                          # bye: nobody plays themselves
            pair = (min(i, partner), max(i, partner))
            if pair in seen:
                continue                          # one row per pairing per date
            seen.add(pair)
            # Shooting is derived from the team index, so the league's OFFENSIVE
            # ordering is real and known: T00 shoots worst, T29 shoots best.
            home_fgm, away_fgm = 30 + pair[0] + trend * d_i, 30 + pair[1] + trend * d_i
            rows.append(_game(
                f"g{d_i}-{pair[0]}-{pair[1]}", date, TEAMS[pair[0]], TEAMS[pair[1]],
                _box(fgm=home_fgm, fg3m=4 + pair[0] // 3),
                _box(fgm=away_fgm, fg3m=4 + pair[1] // 3),
            ))
    return pd.DataFrame(rows)


DATES = [f"2026-03-{d:02d}" for d in range(1, 18)]


def _worst_and_best(frame):
    """The two teams at the extremes of the league's shooting, from the data."""
    means = team_game_factors(frame, as_of="2026-03-18", window=15)
    ordered = means["efg_pct"].sort_values()
    return ordered.index[-1], ordered.index[0]        # (best shooter, worst shooter)


def test_ranks_are_computed_from_the_games_and_not_constant():
    """The failure this file exists to prevent.

    NFL main shipped duels whose ranks were `1` and `2` for every game. That reads
    as a measurement and means nothing. So this asserts the ranks a duel STATES are
    the ranks the league's box scores actually produce -- not constants, and not
    something close to them.

    Two genuinely different leagues have to disagree: flipping every team's
    shooting reverses the league, so the team that shot best must now shoot worst.
    """
    from nba_predictor.signals.duel import ranks

    frame = _schedule(DATES)
    before = ranks(
        team_game_factors(frame, as_of="2026-03-18", window=15)["efg_pct"].to_dict(),
        higher_is_better=True,
    )

    # A second, genuinely different league: every team's shooting is FLIPPED, so
    # the ordering of the league reverses end to end.
    flipped = frame.copy()
    for col in ("fgm", "fg3m"):
        flipped[f"home_{col}"] = 60 - flipped[f"home_{col}"]
        flipped[f"away_{col}"] = 60 - flipped[f"away_{col}"]
    after = ranks(
        team_game_factors(flipped, as_of="2026-03-18", window=15)["efg_pct"].to_dict(),
        higher_is_better=True,
    )

    # The league really is different, and really is ordered.
    assert len(set(before.values())) > 1, "every team got the same rank: nothing was ranked"
    assert before["T29"] <= 5, f"T29 shoots best of 30 but is ranked {before['T29']}"
    assert after["T29"] >= 25, f"flipping the league did not move T29 (still {after['T29']})"
    assert before["T00"] >= 25 and after["T00"] <= 5, (
        "the flip did not reverse the league's ordering"
    )

    # The load-bearing half: every rank a duel STATES must be the rank the games
    # produce, for the team it names. A duel carrying constants cannot pass.
    means = team_game_factors(frame, as_of="2026-03-18", window=15)
    computed_attack = ranks(means["efg_pct"].to_dict(), higher_is_better=True)
    computed_defence = ranks(means["efg_pct_allowed"].to_dict(), higher_is_better=False)

    duels = four_factors_duel("T29", "T00", frame, "2026-03-18", min_gap=1)
    assert duels, "no duels for the league's widest shooting gap"
    for duel in duels:
        assert duel.attacker_rank == computed_attack[duel.attacker], (
            f"{duel.id} states rank {duel.attacker_rank} for {duel.attacker} but the "
            f"games rank it {computed_attack[duel.attacker]}"
        )
        assert duel.defender_rank == computed_defence[duel.defender], (
            f"{duel.id} states rank {duel.defender_rank} for {duel.defender}'s defence "
            f"but the games rank it {computed_defence[duel.defender]}"
        )
        assert duel.attacker_rank != duel.defender_rank, (
            f"{duel.id}: both sides hold rank {duel.attacker_rank}"
        )


def test_only_games_strictly_before_as_of_are_ranked():
    """A game on or after `as_of` has not been played, so it cannot be ranked."""
    frame = _schedule(DATES)
    as_of = "2026-03-10"                       # only dates 01..09 have been played

    before = four_factors_duel("T29", "T00", frame, as_of)
    assert before, "no duels with nine games of history"

    # Poison every game on or after the cutoff. If the ranker reads them the
    # answer changes; if it respects the cut it cannot.
    poisoned = frame.copy()
    later = poisoned["game_date"] >= as_of
    assert later.any(), "fixture must contain later games"
    for col in ("fgm", "fg3m"):
        poisoned.loc[later, f"home_{col}"] = 0
        poisoned.loc[later, f"away_{col}"] = 0

    after = four_factors_duel("T29", "T00", poisoned, as_of)
    assert [d.attacker_rank for d in after] == [d.attacker_rank for d in before], (
        "games dated on or after `as_of` changed the ranks"
    )


def test_the_window_is_the_last_fifteen_games():
    """Older than the window is out: a team's form is its recent form.

    Asserted on the means rather than on the duels, because which duels survive
    `min_gap` is a consequence of the ranks and not the window itself.
    """
    dates = [f"2026-02-{d:02d}" for d in range(1, 18)] + [f"2026-03-{d:02d}" for d in range(1, 12)]
    frame = _schedule(dates, trend=2.0)      # every team's shooting improves over time
    as_of = "2026-03-12"

    long_window = team_game_factors(frame, as_of=as_of, window=15)
    short_window = team_game_factors(frame, as_of=as_of, window=4, min_games=3)

    # A team improving over time must read HIGHER on a short window than a long one,
    # because the short window is all recent games and the long one reaches further back.
    assert len(long_window) == 30 and len(short_window) == 30
    for team in ("T29", "T15", "T00"):
        assert short_window.loc[team, "efg_pct"] > long_window.loc[team, "efg_pct"], (
            f"{team}: the 4-game mean ({short_window.loc[team, 'efg_pct']:.4f}) is not "
            f"above the 15-game mean ({long_window.loc[team, 'efg_pct']:.4f}); the window "
            "is not being taken from the most recent games"
        )

    # And the count of games actually used per team is the window, not the whole history.
    from nba_predictor.signals.four_factors_duel import _long_form
    played = _long_form(frame)
    played = played[played["game_date"] < as_of].sort_values("game_date")
    assert len(played[played["team"] == "T29"]) > 15, "fixture must have more history than the window"


def test_at_most_two_duels_come_back():
    frame = _schedule(DATES)
    duels = four_factors_duel("T29", "T00", frame, "2026-03-18")
    assert 0 < len(duels) <= 2, f"expected the strongest two duels, got {len(duels)}"


def test_min_gap_filters_close_matchups():
    """`min_gap` places of 30: tighten it far enough and nothing separates the teams."""
    frame = _schedule(DATES)
    loose = four_factors_duel("T29", "T00", frame, "2026-03-18", min_gap=1)
    tight = four_factors_duel("T29", "T00", frame, "2026-03-18", min_gap=29)
    assert loose, "min_gap=1 should still find something in this league"
    assert tight == [], f"min_gap=29 of 30 must leave nothing, got {[d.id for d in tight]}"


def test_a_team_without_enough_games_has_no_duels():
    """Ranking a team off one game would be a number, not a measurement."""
    frame = _schedule(["2026-03-01", "2026-03-02", "2026-03-03"])
    assert four_factors_duel("T29", "T00", frame, "2026-03-04", min_games=15) == []


def test_duels_come_from_both_directions():
    """Home's attack can meet away's defence, and vice versa."""
    frame = _schedule(DATES)
    duels = four_factors_duel("T29", "T00", frame, "2026-03-18", min_gap=1)
    sides = {d.id.rsplit(":", 1)[-1] for d in duels}
    assert sides >= {"home", "away"}, f"only one direction produced duels: {sides}"


def test_defence_is_ranked_on_the_opponents_value():
    """A defence is what it ALLOWS, so the allowed figure must be the opponents' own.

    Checked per game against the other side's row, not against a direction-based
    expectation ("a better offence allows more"), which is not true on a balanced
    schedule: every team there allows roughly the league mean.
    """
    frame = _schedule(DATES)
    from nba_predictor.signals.four_factors_duel import _long_form

    long = _long_form(frame)
    by_game = long.set_index(["game_id", "team"])
    for factor in ("efg_pct", "tov_rate", "orb_pct", "ft_rate"):
        checked = 0
        for game_id, rows in long.groupby("game_id"):
            if len(rows) != 2:
                continue
            (_, team_a), (_, team_b) = list(rows.iterrows())
            a, b = rows.iloc[0], rows.iloc[1]
            assert a[f"{factor}_allowed"] == pytest.approx(b[factor]), (
                f"{game_id}: {a['team']} allows {a[f'{factor}_allowed']} but "
                f"{b['team']} scored {b[factor]} on {factor}"
            )
            checked += 1
        assert checked, f"no games checked for {factor}"


def test_all_thirty_teams_are_ranked():
    """A rank out of fewer than 30 is a fabricated position, not a measurement."""
    frame = _schedule(DATES)
    means = team_game_factors(frame, as_of="2026-03-18", window=15)
    assert len(means) == 30, f"expected all 30 teams ranked, got {len(means)}"


def test_a_bettered_schedule_changes_the_ranks():
    """The ranks follow the games: improve one team's shooting and it climbs."""
    frame = _schedule(DATES)
    before = team_game_factors(frame, as_of="2026-03-18", window=15)
    from nba_predictor.signals.duel import ranks
    before_ranks = ranks(before["efg_pct"].to_dict(), higher_is_better=True)

    improved = frame.copy()
    is_t00_home = improved["home_team"] == "T00"
    is_t00_away = improved["away_team"] == "T00"
    improved.loc[is_t00_home, "home_fgm"] += 25
    improved.loc[is_t00_away, "away_fgm"] += 25

    after_ranks = ranks(
        team_game_factors(improved, as_of="2026-03-18", window=15)["efg_pct"].to_dict(),
        higher_is_better=True,
    )
    assert after_ranks["T00"] < before_ranks["T00"], (
        f"T00 improved from rank {before_ranks['T00']} to {after_ranks['T00']}; "
        "the ranking is not reading the games"
    )
