def test_load_hub_cache_returns_empty_list_when_missing(tmp_path):
    from nba_predictor.services.hub_service import load_hub_cache

    assert load_hub_cache(tmp_path / "missing.json") == []


def test_load_hub_cache_reads_json_array(tmp_path):
    import json

    from nba_predictor.services.hub_service import load_hub_cache

    path = tmp_path / "teams.json"
    path.write_text(json.dumps([{"abbreviation": "BOS", "points_per_game": 118.2}]))

    assert load_hub_cache(path) == [{"abbreviation": "BOS", "points_per_game": 118.2}]


def test_load_player_name_map_builds_id_to_name_dict(tmp_path):
    import json

    from nba_predictor.services.hub_service import load_player_name_map

    path = tmp_path / "players.json"
    path.write_text(json.dumps([
        {"player_id": "4251", "player_name": "Paul George"},
        {"player_id": "203999", "player_name": "Nikola Jokic"},
    ]))

    assert load_player_name_map(path) == {"4251": "Paul George", "203999": "Nikola Jokic"}


def test_load_player_name_map_missing_file_returns_empty_dict(tmp_path):
    from nba_predictor.services.hub_service import load_player_name_map

    assert load_player_name_map(tmp_path / "does-not-exist.json") == {}


def test_compute_track_record_groups_by_market(tmp_path):
    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_market_prediction(
        db_path, game_id="g1", market="h2h", selection="home", model_probability=0.6,
        market_probability=0.5, edge=0.1, bookmaker="DraftKings", american_odds=-130,
        created_at="2026-11-01T12:00:00",
    )
    store.insert_market_prediction(
        db_path, game_id="g2", market="h2h", selection="away", model_probability=0.55,
        market_probability=0.5, edge=0.05, bookmaker="DraftKings", american_odds=120,
        created_at="2026-11-02T12:00:00",
    )
    store.insert_market_prediction(
        db_path, game_id="g1", market="spread", selection="home", model_probability=0.52,
        market_probability=0.5, edge=0.02, bookmaker="DraftKings", american_odds=-110,
        created_at="2026-11-01T12:00:00",
    )

    records = compute_track_record(db_path)
    by_market = {r.market: r for r in records}

    assert by_market["h2h"].total_predictions == 2
    assert by_market["spread"].total_predictions == 1


def test_compute_track_record_empty_db_returns_empty_list(tmp_path):
    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    assert compute_track_record(db_path) == []


def _completed_game(game_id, home, away, home_pts, away_pts):
    return {
        "game_id": game_id, "game_date": "2026-03-01", "home_team": home, "away_team": away,
        "completed": True, "home_pts": home_pts, "away_pts": away_pts,
    }


def test_compute_track_record_settles_game_outcome_against_real_results(tmp_path):
    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    # Correct call: predicted BOS (home) to win, BOS did win.
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-03-01T00:00:00", model_version="v1",
        home_win_prob=0.7, predicted_margin=5.0, predicted_total=220.0,
    )
    # Incorrect call: predicted LAL (home) to win, LAL lost.
    store.insert_prediction(
        db_path, game_id="g2", created_at="2026-03-01T00:00:00", model_version="v1",
        home_win_prob=0.6, predicted_margin=3.0, predicted_total=215.0,
    )

    schedule = [
        _completed_game("g1", "BOS", "MIA", 110, 100),
        _completed_game("g2", "LAL", "GSW", 95, 105),
    ]

    records = compute_track_record(db_path, schedule)
    game_outcome = next(r for r in records if r.market == "game_outcome")

    assert game_outcome.total_predictions == 2
    assert game_outcome.correct_predictions == 1
    assert game_outcome.hit_rate == 0.5


def test_compute_track_record_ignores_predictions_for_incomplete_games(tmp_path):
    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-03-01T00:00:00", model_version="v1",
        home_win_prob=0.6, predicted_margin=3.0, predicted_total=220.0,
    )
    schedule = [{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None}]

    records = compute_track_record(db_path, schedule)

    assert not any(r.market == "game_outcome" for r in records)


def test_compute_track_record_settles_h2h_market_predictions(tmp_path):
    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_market_prediction(
        db_path, game_id="g1", market="h2h", selection="BOS", model_probability=0.65,
        market_probability=0.55, edge=0.1, bookmaker="DraftKings", american_odds=-140,
        created_at="2026-03-01T00:00:00",
    )
    schedule = [_completed_game("g1", "BOS", "MIA", 110, 100)]

    records = compute_track_record(db_path, schedule)
    h2h = next(r for r in records if r.market == "h2h")

    assert h2h.total_predictions == 1
    assert h2h.correct_predictions == 1
    assert h2h.hit_rate == 1.0


def test_track_record_counts_a_pick_made_after_tip_off_with_the_pre_tip_subset_beside_it(tmp_path):
    """The old rule, rewritten: g2's backtest row was left out of every rate.

    This test was `test_track_record_counts_only_picks_made_before_tip_off` and
    asserted `total_predictions == 1` / `n_rebuilt == 1` for this fixture: g1
    has a pre-tip pick it got right, g2 has only a backtest row it also got
    right, and g2 was dropped because the model was rerun on it after the game.
    That is the "with every model change it will stop tracking" failure.

    Now both count -- 2/2 -- and the pre-tip subset beside them is the 1/1 the
    old headline showed, with its own n. `n_rebuilt` is no longer the count of
    what was withheld; it is the reconciliation between the two figures.
    """
    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    # g1: a real pre-tip pick (right), later overwritten by a backtest (wrong).
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-03-01T10:00:00+00:00", model_version="v1",
        home_win_prob=0.7, predicted_margin=5.0, predicted_total=220.0,
    )
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-09-20T08:00:00+00:00", model_version="v2",
        home_win_prob=0.3, predicted_margin=-2.0, predicted_total=220.0,
    )
    # g2: only a backtest row, written after the game. Counted now, and never
    # presented as a pre-game pick.
    store.insert_prediction(
        db_path, game_id="g2", created_at="2026-09-20T08:00:00+00:00", model_version="v2",
        home_win_prob=0.4, predicted_margin=-3.0, predicted_total=215.0,
    )
    schedule = [
        _completed_game("g1", "BOS", "MIA", 110, 100),
        _completed_game("g2", "LAL", "GSW", 95, 105),
    ]

    game_outcome = next(r for r in compute_track_record(db_path, schedule) if r.market == "game_outcome")

    assert game_outcome.total_predictions == 2
    assert game_outcome.correct_predictions == 2
    # The pre-tip subset is the figure the old headline published.
    assert game_outcome.n_pre_tip == 1
    assert game_outcome.pre_tip.total_predictions == 1
    assert game_outcome.pre_tip.correct_predictions == 1
    assert game_outcome.n_rebuilt == 1
    assert game_outcome.total_predictions == game_outcome.pre_tip.total_predictions + game_outcome.n_rebuilt
    # Disclosure is per pick, and g1's counted pick is the pre-tip one even
    # though the backtest row for it is the better guess.
    assert {p.game_id: p.made_before_tip for p in game_outcome.per_pick if p.counted} == {
        "g1": True, "g2": False,
    }


def test_track_record_h2h_counts_a_market_row_written_after_tip_off(tmp_path):
    """The old rule, rewritten: this market row was graded by nobody.

    It was `test_track_record_h2h_ignores_market_rows_made_after_tip_off` and
    asserted `total_predictions == 0` for a single h2h row written after
    tip-off: the whole market was absent from the record because a model change
    had been run on it. Now the row is the counted pick for that (game, market),
    the verdict is graded, and it is labelled not-pre-tip.
    """
    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_market_prediction(
        db_path, game_id="g1", market="h2h", selection="MIA", model_probability=0.6,
        market_probability=0.5, edge=0.1, bookmaker="DraftKings", american_odds=120,
        created_at="2026-03-02T03:00:00+00:00",
    )
    schedule = [_completed_game("g1", "BOS", "MIA", 110, 100)]

    h2h = next(r for r in compute_track_record(db_path, schedule) if r.market == "h2h")

    assert h2h.total_predictions == 1
    assert h2h.correct_predictions == 0, "MIA was picked and BOS won"
    assert h2h.n_pre_tip == 0
    assert h2h.pre_tip.total_predictions == 0
    assert h2h.per_pick[0].made_before_tip is False


def test_track_record_settles_spread_picks_against_the_closing_line(tmp_path):
    """Spread rows are judged against the line the pick was priced at.

    `point` is the selection's own line in book convention (BOS -4.5 stores
    -4.5 for BOS), so the side covers when its margin beats `-point`.
    """
    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    # Model's side: BOS -4.5, BOS won by 10 -> cleared the line, correct.
    store.insert_market_prediction(
        db_path, game_id="g1", market="spread", selection="BOS", model_probability=0.6,
        market_probability=0.5, edge=0.1, bookmaker="DraftKings", american_odds=-110,
        created_at="2026-03-01T00:00:00", point=-4.5,
    )
    # Model's side: LAL +3.5, LAL lost by 10 -> did not cover, wrong.
    store.insert_market_prediction(
        db_path, game_id="g2", market="spread", selection="LAL", model_probability=0.55,
        market_probability=0.5, edge=0.05, bookmaker="DraftKings", american_odds=-110,
        created_at="2026-03-01T00:00:00", point=3.5,
    )
    schedule = [
        _completed_game("g1", "BOS", "MIA", 110, 100),  # margin +10
        _completed_game("g2", "LAL", "GSW", 95, 105),   # margin -10
    ]

    spread = next(r for r in compute_track_record(db_path, schedule) if r.market == "spread")

    assert (spread.total_predictions, spread.correct_predictions, spread.hit_rate) == (2, 1, 0.5)
    assert spread.settled is True


def test_spread_grading_uses_the_same_threshold_as_the_odds_writer(tmp_path):
    """The writer prices a cover with `line=-point` (refresh_odds.
    _model_probability); the grader must clear the SAME threshold, or the
    record would judge every pick by a rule the pick was never priced with.
    The convention is pinned against `normal_cover_probability` rather than
    restated, because a grader written from the opposite reading produces
    plausible-looking rates that are all mirrored."""
    from nba_predictor.odds.value_bets import normal_cover_probability
    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    # The writer's view for BOS -4.5 with a 10-point predicted margin.
    assert normal_cover_probability(mean=10.0, line=4.5, std=12.0) > 0.5

    for game_id, home_pts, away_pts in (("g1", 114, 100), ("g2", 104, 100)):
        store.insert_market_prediction(
            db_path, game_id=game_id, market="spread", selection="BOS", model_probability=0.6,
            market_probability=0.5, edge=0.1, bookmaker="DraftKings", american_odds=-110,
            created_at="2026-03-01T00:00:00", point=-4.5,
        )
    schedule = [
        _completed_game("g1", "BOS", "MIA", 114, 100),  # margin 14 > 4.5 -> covered
        _completed_game("g2", "BOS", "MIA", 104, 100),  # margin 4 < 4.5 -> did not cover
    ]

    spread = next(r for r in compute_track_record(db_path, schedule) if r.market == "spread")

    assert (spread.total_predictions, spread.correct_predictions) == (2, 1)


def test_track_record_settles_total_picks_against_the_line(tmp_path):
    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_market_prediction(
        db_path, game_id="g1", market="total", selection="over", model_probability=0.6,
        market_probability=0.5, edge=0.1, bookmaker="DraftKings", american_odds=-110,
        created_at="2026-03-01T00:00:00", point=205.5,
    )
    store.insert_market_prediction(
        db_path, game_id="g2", market="total", selection="under", model_probability=0.55,
        market_probability=0.5, edge=0.05, bookmaker="DraftKings", american_odds=-110,
        created_at="2026-03-01T00:00:00", point=214.5,
    )
    schedule = [
        _completed_game("g1", "BOS", "MIA", 110, 100),  # total 210 > 205.5 -> over wins
        _completed_game("g2", "LAL", "GSW", 105, 105),  # total 210 < 214.5 -> under wins
    ]

    total = next(r for r in compute_track_record(db_path, schedule) if r.market == "total")

    assert (total.total_predictions, total.correct_predictions, total.hit_rate) == (2, 2, 1.0)


def test_track_record_counts_a_push_as_neither_right_nor_wrong(tmp_path):
    """A final margin that lands exactly on the line is a push: nobody won it.
    It is counted in `n_push` and left out of the rate -- counting it as a
    miss would make a 0.0-line bet look like a wrong pick."""
    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_market_prediction(
        db_path, game_id="g1", market="spread", selection="BOS", model_probability=0.6,
        market_probability=0.5, edge=0.1, bookmaker="DraftKings", american_odds=-110,
        created_at="2026-03-01T00:00:00", point=-4.0,
    )
    schedule = [_completed_game("g1", "BOS", "MIA", 104, 100)]  # margin 4 == the line

    spread = next(r for r in compute_track_record(db_path, schedule) if r.market == "spread")

    assert spread.total_predictions == 0
    assert spread.correct_predictions == 0
    assert spread.n_push == 1
    # No graded pick means no rate: None, never 0.0 (0% claims every pick missed).
    assert spread.hit_rate is None


def test_track_record_marks_markets_it_cannot_judge_as_unsettled(tmp_path):
    """Rows for a market this repo has no rule for are shown with their stored
    count and no rate -- never a fabricated 0%."""
    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_market_prediction(
        db_path, game_id="g9", market="player_points", selection="Jayson Tatum",
        model_probability=0.6, market_probability=None, edge=None, bookmaker=None,
        american_odds=None, created_at="2026-03-01T00:00:00",
    )

    row = next(r for r in compute_track_record(db_path, []) if r.market == "player_points")

    assert row.settled is False
    assert row.total_predictions == 1
    assert row.hit_rate is None


def _game_on(game_id, home, away, home_pts, away_pts, game_date):
    return {
        "game_id": game_id, "game_date": game_date, "home_team": home, "away_team": away,
        "completed": True, "home_pts": home_pts, "away_pts": away_pts,
    }


def test_track_record_weekly_fills_every_week_since_tracking_began(tmp_path):
    """A week with no graded picks still gets a row: tracked=false, rate None.

    Grouping the picks by week and emitting one row per group would make a
    week the tracker skipped read as if it never existed. The window runs
    from the first game the tracker wrote a pick for through the current
    week, so the gap is visible as a gap.
    """
    from datetime import date

    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    # g1 tips in the week of Mon 2026-03-02, g3 in the week of Mon 2026-03-16.
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-03-02T00:00:00", model_version="v1",
        home_win_prob=0.7, predicted_margin=5.0, predicted_total=220.0,
    )
    store.insert_prediction(
        db_path, game_id="g3", created_at="2026-03-16T00:00:00", model_version="v1",
        home_win_prob=0.6, predicted_margin=3.0, predicted_total=215.0,
    )
    schedule = [
        _game_on("g1", "BOS", "MIA", 110, 100, "2026-03-02"),  # correct (home won)
        _game_on("g3", "LAL", "GSW", 95, 105, "2026-03-16"),   # wrong (home lost)
    ]

    row = next(r for r in compute_track_record(db_path, schedule, today=date(2026, 3, 20))
               if r.market == "game_outcome")
    weeks = {w.week_start: w for w in row.weekly}

    assert list(weeks) == ["2026-03-02", "2026-03-09", "2026-03-16"]
    assert weeks["2026-03-02"].tracked is True
    assert (weeks["2026-03-02"].n, weeks["2026-03-02"].correct, weeks["2026-03-02"].hit_rate) == (1, 1, 1.0)
    # The gap week: a count of nothing, and no rate at all. 0.0 would claim
    # the model was wrong on every pick it never made.
    assert weeks["2026-03-09"].tracked is False
    assert (weeks["2026-03-09"].n, weeks["2026-03-09"].correct) == (0, 0)
    assert weeks["2026-03-09"].hit_rate is None
    assert weeks["2026-03-16"].tracked is True
    assert (weeks["2026-03-16"].n, weeks["2026-03-16"].correct, weeks["2026-03-16"].hit_rate) == (1, 0, 0.0)


def test_track_record_weekly_reaches_the_current_week(tmp_path):
    from datetime import date

    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-03-02T00:00:00", model_version="v1",
        home_win_prob=0.7, predicted_margin=5.0, predicted_total=220.0,
    )
    schedule = [_game_on("g1", "BOS", "MIA", 110, 100, "2026-03-02")]

    row = next(r for r in compute_track_record(db_path, schedule, today=date(2026, 3, 25))
               if r.market == "game_outcome")

    # Through the current week (Mon 2026-03-23), not through the last pick.
    assert [w.week_start for w in row.weekly] == ["2026-03-02", "2026-03-09", "2026-03-16", "2026-03-23"]
    assert row.weekly[-1].tracked is False
    assert row.weekly[-1].hit_rate is None


def test_track_record_weekly_sums_to_the_headline(tmp_path):
    """The identity the panel reconciles with: the week rows add up to the
    headline above them, pick for pick and hit for hit. If they can drift,
    the page shows two counts of the same record that disagree."""
    from datetime import date

    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-03-02T00:00:00", model_version="v1",
        home_win_prob=0.7, predicted_margin=5.0, predicted_total=220.0,
    )
    store.insert_prediction(
        db_path, game_id="g2", created_at="2026-03-03T00:00:00", model_version="v1",
        home_win_prob=0.6, predicted_margin=3.0, predicted_total=215.0,
    )
    store.insert_market_prediction(
        db_path, game_id="g1", market="h2h", selection="BOS", model_probability=0.65,
        market_probability=0.55, edge=0.1, bookmaker="DraftKings", american_odds=-140,
        created_at="2026-03-02T00:00:00",
    )
    store.insert_market_prediction(
        db_path, game_id="g2", market="spread", selection="LAL", model_probability=0.6,
        market_probability=0.5, edge=0.1, bookmaker="DraftKings", american_odds=-110,
        created_at="2026-03-03T00:00:00", point=-4.5,
    )
    schedule = [
        _game_on("g1", "BOS", "MIA", 110, 100, "2026-03-02"),
        _game_on("g2", "LAL", "GSW", 95, 105, "2026-03-03"),
    ]

    records = compute_track_record(db_path, schedule, today=date(2026, 3, 8))
    settled = [r for r in records if r.settled]

    assert {r.market for r in settled} == {"game_outcome", "h2h", "spread"}
    for row in settled:
        assert sum(w.n for w in row.weekly) == row.total_predictions, row.market
        assert sum(w.correct for w in row.weekly) == row.correct_predictions, row.market


def test_track_record_weekly_windows_align_across_markets(tmp_path):
    """Every settled market's week list covers the same weeks in the same
    order, so the panel can lay them side by side in one table without
    inventing a join or a missing row."""
    from datetime import date

    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-03-16T00:00:00", model_version="v1",
        home_win_prob=0.7, predicted_margin=5.0, predicted_total=220.0,
    )
    store.insert_market_prediction(
        db_path, game_id="g1", market="h2h", selection="BOS", model_probability=0.65,
        market_probability=0.55, edge=0.1, bookmaker="DraftKings", american_odds=-140,
        created_at="2026-03-16T00:00:00",
    )
    schedule = [_game_on("g1", "BOS", "MIA", 110, 100, "2026-03-16")]

    records = compute_track_record(db_path, schedule, today=date(2026, 3, 22))
    weeks = {r.market: [w.week_start for w in r.weekly] for r in records if r.settled}

    assert weeks["game_outcome"] == weeks["h2h"]


def test_track_record_weekly_is_empty_when_nothing_was_graded(tmp_path):
    """Stored rows with no results to judge them against carry no week rows
    either -- an empty weekly list says 'nothing measured' where a list of
    untracked weeks would imply a window the tracker never had."""
    from datetime import date

    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_market_prediction(
        db_path, game_id="g9", market="h2h", selection="BOS", model_probability=0.6,
        market_probability=0.5, edge=0.1, bookmaker="DraftKings", american_odds=-140,
        created_at="2026-03-01T00:00:00",
    )
    # The game has not finished: nothing to grade, and no window to lay out.
    schedule = [{
        "game_id": "g9", "game_date": "2026-03-01", "home_team": "BOS", "away_team": "MIA",
        "completed": False, "home_pts": None, "away_pts": None,
    }]

    records = compute_track_record(db_path, schedule, today=date(2026, 3, 8))

    assert records and all(r.weekly == [] for r in records)


def test_track_record_h2h_settles_one_model_pick_per_game(tmp_path):
    """Odds refresh stores both sides for every bookmaker on every run; the
    record judges only the model's side, once per game, from the latest
    pre-tip run."""
    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    for created_at in ("2026-03-01T09:00:00+00:00", "2026-03-01T10:00:00+00:00"):
        for book in ("DraftKings", "FanDuel"):
            for selection, prob in (("BOS", 0.65), ("MIA", 0.35)):
                store.insert_market_prediction(
                    db_path, game_id="g1", market="h2h", selection=selection, model_probability=prob,
                    market_probability=0.5, edge=prob - 0.5, bookmaker=book, american_odds=-110,
                    created_at=created_at,
                )
    # A post-tip run that flips the pick must be ignored.
    store.insert_market_prediction(
        db_path, game_id="g1", market="h2h", selection="MIA", model_probability=0.9,
        market_probability=0.5, edge=0.4, bookmaker="DraftKings", american_odds=-110,
        created_at="2026-03-02T03:00:00+00:00",
    )
    schedule = [_completed_game("g1", "BOS", "MIA", 110, 100)]

    h2h = next(r for r in compute_track_record(db_path, schedule) if r.market == "h2h")

    assert (h2h.total_predictions, h2h.correct_predictions) == (1, 1)
