from datetime import datetime, timezone


def test_pick_cutoff_uses_tip_off_when_known():
    from nba_predictor.tracking.timing import pick_cutoff

    game = {"game_date": "2026-03-01", "tip_off": "2026-03-02T00:30Z"}
    assert pick_cutoff(game) == datetime(2026, 3, 2, 0, 30, tzinfo=timezone.utc)


def test_pick_cutoff_falls_back_to_noon_eastern_on_the_game_date():
    from nba_predictor.tracking.timing import pick_cutoff

    # EST in winter (UTC-5), EDT in spring (UTC-4).
    assert pick_cutoff({"game_date": "2026-01-10"}) == datetime(2026, 1, 10, 17, 0, tzinfo=timezone.utc)
    assert pick_cutoff({"game_date": "2026-04-10"}) == datetime(2026, 4, 10, 16, 0, tzinfo=timezone.utc)


def test_made_before_tip_compares_in_utc_and_reads_zoneless_as_utc():
    from nba_predictor.tracking.timing import made_before_tip

    game = {"game_date": "2026-03-01", "tip_off": "2026-03-02T00:30Z"}
    assert made_before_tip("2026-03-01T23:00:00+00:00", game) is True
    assert made_before_tip("2026-03-01T23:00:00", game) is True
    assert made_before_tip("2026-03-02T00:30:00+00:00", game) is False
    # A backtest written months later is never pre-tip.
    assert made_before_tip("2026-09-20T08:00:00+00:00", game) is False


def test_made_before_tip_never_counts_an_unreadable_timestamp():
    from nba_predictor.tracking.timing import made_before_tip

    assert made_before_tip("not a date", {"game_date": "2026-03-01"}) is False
    assert made_before_tip("2026-03-01T00:00:00", {"game_date": "garbage"}) is False
