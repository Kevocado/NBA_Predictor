def test_missing_players_value_sums_across_players():
    from nba_predictor.features.injuries import missing_players_value

    def fake_value_fn(player_id, season):
        return {101: 5.0, 102: 3.0}[player_id]

    total = missing_players_value([101, 102], season=2026, value_fn=fake_value_fn)
    assert total == 8.0


def test_missing_players_value_empty_list_is_zero():
    from nba_predictor.features.injuries import missing_players_value

    assert missing_players_value([], season=2026) == 0.0


def test_missing_players_value_uses_real_data_module_by_default(monkeypatch):
    from nba_predictor.features import injuries as injuries_feature
    from nba_predictor.data import injuries as injuries_data

    monkeypatch.setattr(injuries_data, "get_missing_player_value", lambda player_id, season: 2.5)

    total = injuries_feature.missing_players_value([1, 2, 3], season=2026)
    assert total == 7.5
