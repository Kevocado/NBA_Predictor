"""Model against the price on the moneyline: /hub/vs-market.

The NBA store already holds what this needs: per bookmaker, after the Shin
de-vig, `market_probability` for both sides plus the model's own probability
for its side (pipeline/refresh_odds). So the comparison needs no implied-
probability conversion of a line -- it reads the price the pick was measured
against, which is the number the edge column was already computed from.
"""
from datetime import date


def _completed_game(game_id, home, away, home_pts, away_pts, game_date="2026-03-01"):
    return {
        "game_id": game_id, "game_date": game_date, "home_team": home, "away_team": away,
        "completed": True, "home_pts": home_pts, "away_pts": away_pts,
    }


def _h2h(db_path, game_id, selection, model, market, created_at="2026-03-01T00:00:00"):
    from nba_predictor.tracking import store

    store.insert_market_prediction(
        db_path, game_id=game_id, market="h2h", selection=selection,
        model_probability=model, market_probability=market,
        edge=None if market is None else model - market,
        bookmaker="DraftKings", american_odds=-110, created_at=created_at,
    )


def _vs_market(db_path, schedule, today=date(2026, 3, 8)):
    from nba_predictor.services.hub_service import compute_vs_market

    return compute_vs_market(db_path, schedule, today=today)


def test_vs_market_scores_only_finished_games_with_a_price(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    # g1: finished, both sides priced -> comparable.
    _h2h(db_path, "g1", "BOS", 0.65, 0.70)
    _h2h(db_path, "g1", "MIA", 0.35, 0.30)
    # g2: finished, but the book never priced it -> nothing to compare with.
    _h2h(db_path, "g2", "LAL", 0.55, None)
    _h2h(db_path, "g2", "GSW", 0.45, None)
    # g3: priced, but not finished -> no result to judge agreement against.
    _h2h(db_path, "g3", "NYK", 0.60, 0.62)
    _h2h(db_path, "g3", "BKN", 0.40, 0.38)
    schedule = [
        _completed_game("g1", "BOS", "MIA", 110, 100),
        _completed_game("g2", "LAL", "GSW", 95, 105),
        {"game_id": "g3", "game_date": "2026-03-01", "home_team": "NYK", "away_team": "BKN",
         "completed": False, "home_pts": None, "away_pts": None},
    ]

    vs = _vs_market(db_path, schedule)

    assert vs.market == "h2h"
    assert vs.n == 1


def test_vs_market_edge_is_model_minus_price_in_points(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    _h2h(db_path, "g1", "BOS", 0.65, 0.55)
    _h2h(db_path, "g1", "MIA", 0.35, 0.45)
    schedule = [_completed_game("g1", "BOS", "MIA", 110, 100)]

    vs = _vs_market(db_path, schedule)

    # The model's side: 65% model, 55% price -> +10 points, in points not
    # probability points, and signed toward the model.
    assert vs.n == 1
    assert vs.mean_model_probability == 0.65
    assert vs.mean_market_probability == 0.55
    assert vs.mean_edge_points == 10.0


def test_vs_market_disagreement_cohort_is_the_picks_against_the_price(tmp_path):
    """Cohort = games where the model's side was NOT the side the price
    favoured, and its hit rate over exactly those games."""
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    # g1: model and price both like BOS -> agreement, not in the cohort.
    _h2h(db_path, "g1", "BOS", 0.65, 0.70)
    _h2h(db_path, "g1", "MIA", 0.35, 0.30)
    # g2: the model backed GSW at a price that fancied LAL -> disagreement,
    # and GSW won, so the cohort's hit rate has one hit in one game.
    _h2h(db_path, "g2", "GSW", 0.55, 0.40)
    _h2h(db_path, "g2", "LAL", 0.45, 0.60)
    schedule = [
        _completed_game("g1", "BOS", "MIA", 110, 100),
        _completed_game("g2", "LAL", "GSW", 95, 105),
    ]

    vs = _vs_market(db_path, schedule)

    assert vs.n == 2
    # (-5 + 15) / 2, averaged over the two comparable games.
    assert vs.mean_edge_points == 5.0
    assert vs.disagreement_n == 1
    assert vs.disagreement_hit_rate == 1.0
    assert vs.disagreement_game_ids == ["g2"]


def test_vs_market_an_even_price_favours_nobody(tmp_path):
    """A 50/50 price is not agreement and not disagreement: the game is still
    compared (it has a price), it just has no side to take a stand against."""
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    _h2h(db_path, "g1", "BOS", 0.60, 0.50)
    _h2h(db_path, "g1", "MIA", 0.40, 0.50)
    schedule = [_completed_game("g1", "BOS", "MIA", 110, 100)]

    vs = _vs_market(db_path, schedule)

    assert vs.n == 1
    assert vs.mean_edge_points == 10.0
    assert vs.disagreement_n == 0
    assert vs.disagreement_hit_rate is None


def test_vs_market_scope_reconciles_the_headline_with_the_week_table(tmp_path):
    """n_total == n_in_weekly + n_outside_weekly, with n_in_weekly read off
    the weekly rows themselves so the note cannot drift from the table it
    describes. Asserted, not assumed: this is the number that tells a reader
    why two counts on one page differ."""
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    _h2h(db_path, "g1", "BOS", 0.65, 0.55)
    _h2h(db_path, "g1", "MIA", 0.35, 0.45)
    _h2h(db_path, "g2", "NYK", 0.58, 0.52, created_at="2026-03-03T00:00:00")
    _h2h(db_path, "g2", "BKN", 0.42, 0.48, created_at="2026-03-03T00:00:00")
    schedule = [
        _completed_game("g1", "BOS", "MIA", 110, 100),
        _completed_game("g2", "NYK", "BKN", 101, 99, game_date="2026-03-03"),
    ]

    vs = _vs_market(db_path, schedule, today=date(2026, 3, 8))
    scope = vs.scope

    assert scope.n_games_total == 2
    assert scope.n_games_in_weekly == sum(w.n for w in vs.weekly)
    assert scope.n_games_total == scope.n_games_in_weekly + scope.n_games_outside_weekly
    # g1 tips on Sun 2026-03-01 (week of Mon 2026-02-23); g2 on Tue
    # 2026-03-03; today (2026-03-08) is itself in the week of 2026-03-02.
    assert scope.weekly_from == "2026-02-23"
    assert scope.weekly_through == "2026-03-02"


def test_vs_market_weekly_gap_reads_as_not_tracked(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    # Week of Mon 2026-03-02, then nothing until the week of Mon 2026-03-16.
    _h2h(db_path, "g1", "BOS", 0.65, 0.55, created_at="2026-03-02T00:00:00")
    _h2h(db_path, "g1", "MIA", 0.35, 0.45, created_at="2026-03-02T00:00:00")
    _h2h(db_path, "g2", "NYK", 0.58, 0.52, created_at="2026-03-16T00:00:00")
    _h2h(db_path, "g2", "BKN", 0.42, 0.48, created_at="2026-03-16T00:00:00")
    schedule = [
        _completed_game("g1", "BOS", "MIA", 110, 100, game_date="2026-03-02"),
        _completed_game("g2", "NYK", "BKN", 101, 99, game_date="2026-03-16"),
    ]

    vs = _vs_market(db_path, schedule, today=date(2026, 3, 20))
    weeks = {w.week_start: w for w in vs.weekly}

    assert list(weeks) == ["2026-03-02", "2026-03-09", "2026-03-16"]
    gap = weeks["2026-03-09"]
    tracked, n, edge = gap.tracked, gap.n, gap.mean_edge_points
    assert tracked is False
    assert n == 0
    # No comparison happened: no rate, no edge. 0.0 pt would claim the model
    # and the price agreed exactly, which was never measured.
    assert edge is None


def test_vs_market_method_sentences_are_in_the_payload(tmp_path):
    """The page prints these verbatim. A number nobody can interpret is not a
    decision aid, and a disclaimer nobody reads is not one either."""
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    _h2h(db_path, "g1", "BOS", 0.65, 0.55)
    _h2h(db_path, "g1", "MIA", 0.35, 0.45)
    schedule = [_completed_game("g1", "BOS", "MIA", 110, 100)]

    vs = _vs_market(db_path, schedule)

    for key in ("edge", "disagreement", "not_a_profit_claim", "population"):
        assert key in vs.method, key
        assert len(str(vs.method[key])) > 40, key
    # The no-profit framing is the brand rule, in the payload not the page.
    assert "profit" in str(vs.method["not_a_profit_claim"]).lower()
    assert "return" in str(vs.method["not_a_profit_claim"]).lower()


def test_vs_market_empty_db_compares_nothing(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    vs = _vs_market(db_path, [])

    assert vs.n == 0
    assert vs.mean_edge_points is None
    assert vs.disagreement_n == 0
    assert vs.disagreement_hit_rate is None
    assert vs.weekly == []
    assert vs.scope.n_games_total == 0
    # The method block is present even when the numbers are not: the page
    # still has to be able to say how the comparison would be made.
    assert "not_a_profit_claim" in vs.method
