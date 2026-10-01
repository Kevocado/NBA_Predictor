def test_missing_players_value_sums_across_players():
    from nba_predictor.features.injuries import missing_players_value

    def fake_value_fn(player_id, season):
        return {101: 5.0, 102: 3.0}[player_id]

    total = missing_players_value([101, 102], season=2026, value_fn=fake_value_fn)
    assert total == 8.0


def test_missing_players_value_empty_list_is_zero():
    from nba_predictor.features.injuries import missing_players_value

    assert missing_players_value([], season=2026) == 0.0


def test_missing_players_value_has_no_default_data_path():
    # This test used to monkeypatch data/injuries.py and assert the feature
    # summed 2.5 per player through it -- i.e. it asserted that the stub WAS
    # the default source. That stub returned {} from _fetch_player_stats and
    # 0.0 per player, so the number it produced was fabricated while reading as
    # measured. It is now neutralised (see tests/test_data_injuries.py) and the
    # feature has no default at all: a caller with real values passes a
    # value_fn, and a caller without one is told so rather than handed an
    # invented total. Assert the refusal instead.
    import pytest

    from nba_predictor.features import injuries as injuries_feature

    with pytest.raises(NotImplementedError):
        injuries_feature.missing_players_value([1, 2, 3], season=2026)
