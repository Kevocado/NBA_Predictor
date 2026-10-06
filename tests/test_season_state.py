"""Season state, and the playoff labels it makes false (spec section 10, G8).

`compute_standings` labels a team "clinched" or "eliminated" purely from its
seed in the conference. During the off-season the completed-games window is
either empty or entirely last season, and both cases produce confident labels
about a season that has not started: six teams "clinched" because they happened
to lead an empty table, or the leader "eliminated" because only one team has a
record at all.

The fix is to derive the season's phase from the schedule itself -- the one
place that knows whether there is a season to be in -- and to suppress the label
when there is not. Not by softening the wording: by not asserting it.
"""

from datetime import date

from nba_predictor.pipeline.ingest import compute_standings, season_state

TODAY = date(2026, 10, 4)


def game(game_id, game_date, home="BOS", away="MIA", home_pts=None, away_pts=None):
    """A schedule row. A finished game also carries `completed` and one box
    field, which is what `compute_standings` filters on -- without those the
    standings come back empty and a test about playoff labels would pass for
    the wrong reason."""
    row = {"game_id": game_id, "game_date": game_date, "home_team": home, "away_team": away}
    if home_pts is not None:
        row.update(completed=True, home_pts=home_pts, away_pts=away_pts, home_fgm=40)
    return row


# --- season_state -----------------------------------------------------------

def test_a_schedule_with_nothing_in_it_is_the_offseason():
    assert season_state([], today=TODAY) == "offseason"


def test_only_last_seasons_games_left_in_the_window_is_the_offseason():
    """The real case: the refresh window still holds last season's finished
    games, none of them current, and nothing scheduled forward."""
    last_season = [
        game("g1", "2026-04-12", home_pts=110, away_pts=98),
        game("g2", "2026-04-14", home_pts=101, away_pts=105),
    ]
    assert season_state(last_season, today=TODAY) == "offseason"


def test_games_scheduled_forward_with_no_results_yet_is_the_preseason():
    upcoming = [game("u1", "2026-10-21"), game("u2", "2026-10-23")]
    assert season_state(upcoming, today=TODAY) == "preseason"


def test_results_and_games_both_present_is_the_regular_season():
    mixed = [
        game("g1", "2026-10-21", home_pts=110, away_pts=98),
        game("u1", "2026-10-25"),
    ]
    assert season_state(mixed, today=TODAY) == "regular"


def test_results_but_nothing_scheduled_forward_is_the_postseason():
    """A season that ended within the gap. `today` sits just after the game, so
    the schedule reads as "finished, nothing ahead" -- the same shape an
    off-season window has, told apart only by how long ago."""
    finished = [game("g1", "2026-06-11", home_pts=110, away_pts=98)]
    assert season_state(finished, today=date(2026, 6, 20)) == "postseason"


def test_the_same_shape_stale_enough_past_is_the_offseason():
    """The case the schedule cannot settle alone, and the reason `OFFSEASON_GAP_DAYS`
    exists: identical rows, four months later."""
    finished = [game("g1", "2026-06-11", home_pts=110, away_pts=98)]
    assert season_state(finished, today=TODAY) == "offseason"


def test_a_game_today_counts_as_scheduled_not_as_a_result():
    """"Today" is ambiguous at the edges, and getting it wrong on the opening day
    labels the whole league "preseason". A game dated today with no score is
    upcoming."""
    assert season_state([game("g1", "2026-10-04")], today=TODAY) == "preseason"


# --- the labels it suppresses ----------------------------------------------

# Six-plus teams in one conference, so seeds past 6 exist and would otherwise
# be labelled. All games finished and all dated in the past.
LAST_SEASON = [
    game("a1", "2026-04-10", home="BOS", away="MIA", home_pts=110, away_pts=98),
    game("a2", "2026-04-10", home="MIA", away="NYK", home_pts=101, away_pts=105),
    game("a3", "2026-04-11", home="LAL", away="DEN", home_pts=99, away_pts=115),
    game("a4", "2026-04-11", home="DEN", away="GSW", home_pts=104, away_pts=100),
    game("a5", "2026-04-12", home="BOS", away="LAL", home_pts=88, away_pts=120),
    game("a6", "2026-04-12", home="NYK", away="GSW", home_pts=90, away_pts=112),
    game("a7", "2026-04-13", home="MIA", away="DEN", home_pts=118, away_pts=99),
]


def test_offseason_standings_carry_no_playoff_status():
    """The bug, stated as a test: last season's window made six teams "clinched"
    and one "eliminated" about a season that has not started."""
    assert season_state(LAST_SEASON, today=TODAY) == "offseason"
    standings = compute_standings(LAST_SEASON, today=TODAY)
    assert standings, "the standings themselves are still real"
    assert all(row["playoff_status"] is None for row in standings), (
        f"off-season standings claimed a playoff race: "
        f"{[r['playoff_status'] for r in standings]}"
    )


def test_a_preseason_standings_table_is_also_unlabelled():
    upcoming = LAST_SEASON + [game("u1", "2026-10-21", home="BOS", away="DEN")]
    assert season_state(upcoming, today=TODAY) == "preseason"
    assert all(r["playoff_status"] is None for r in compute_standings(upcoming, today=TODAY))


def test_the_regular_season_still_labelled():
    """Suppressing everything would be as useless as labelling everything."""
    assert season_state(LAST_SEASON, today=date(2026, 4, 15)) == "postseason"
    mid = LAST_SEASON + [game("u1", "2026-04-20", home="BOS", away="DEN")]
    standings = compute_standings(mid, today=date(2026, 4, 15))
    assert any(r["playoff_status"] == "clinched" for r in standings), (
        "a season in progress must still say who has clinched"
    )


def test_the_state_is_reported_alongside_the_standings():
    """Callers need the phase to label the page; deriving it twice invites the
    two to disagree."""
    rows = compute_standings(LAST_SEASON, today=TODAY)
    assert rows[0]["season_state"] == "offseason"


def test_an_empty_schedule_yields_no_standings_and_still_names_the_state():
    assert compute_standings([], today=TODAY) == []


def test_the_real_october_window_is_preseason_not_a_race():
    """The production shape, and the one that produced the bug.

    The rolling ingest window holds last season's finished games all summer and
    the new season's upcoming schedule. Both "there are results" and "there are
    games ahead" are true, so the naive reading is `regular` -- and then six
    teams are "clinched" about a league nobody has played in.
    """
    window = LAST_SEASON + [
        game("u1", "2026-10-21", home="BOS", away="DEN"),
        game("u2", "2026-10-23", home="MIA", away="GSW"),
    ]
    assert season_state(window, today=TODAY) == "preseason"
    assert all(r["playoff_status"] is None for r in compute_standings(window, today=TODAY))


def test_one_played_game_this_season_makes_it_a_race_again():
    """The same window, one week later, after opening night. The gate has to open
    as well as close, or the league never gets a playoff race."""
    opened = LAST_SEASON + [
        game("n1", "2026-10-01", home="BOS", away="DEN", home_pts=110, away_pts=98),
        game("u1", "2026-10-21", home="MIA", away="GSW"),
    ]
    assert season_state(opened, today=TODAY) == "regular"
    assert any(r["playoff_status"] for r in compute_standings(opened, today=TODAY))
