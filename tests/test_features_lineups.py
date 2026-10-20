"""Player minutes and starting-five projections (availability B3, offline).

The module is evaluated against real box scores in tools/evaluate_lineups.py.
These tests pin the arithmetic and the two rules that are easy to get wrong:
the window is the team's last ten games, and only fully available teammates
absorb a removed player's minutes.
"""

from __future__ import annotations

import gzip
import json

import pytest

from nba_predictor.features import lineups


def _row(pid, name, team, minutes, *, started=False, position="G"):
    return {
        "player_id": str(pid),
        "player_name": name,
        "team": team,
        "minutes": minutes,
        "started": started,
        "position": position,
    }


def _game(gid, rows):
    return {"game_id": gid, "players": rows}


def _three_games(team="BOS"):
    return [
        _game("g1", [
            _row(1, "One", team, 30, started=True),
            _row(2, "Two", team, 20, started=True),
            _row(3, "Three", team, 10),
        ]),
        _game("g2", [
            _row(1, "One", team, 20, started=True),
            _row(2, "Two", team, 10, started=True),
            _row(3, "Three", team, 30),
        ]),
        _game("g3", [
            _row(1, "One", team, 40, started=True),
            _row(2, "Two", team, 0),
            _row(3, "Three", team, 20, started=True),
        ]),
    ]


def _by_id(projections):
    return {p.player_id: p for p in projections}


def test_ewma_minutes_and_start_probability_use_every_team_game():
    projections = lineups.project_team("BOS", _three_games())

    p = _by_id(projections)
    alpha = 2 / 11
    one = 40 * alpha + (20 * alpha + 30 * (1 - alpha)) * (1 - alpha)
    two = 0 * alpha + (10 * alpha + 20 * (1 - alpha)) * (1 - alpha)
    three = 20 * alpha + (30 * alpha + 10 * (1 - alpha)) * (1 - alpha)
    assert p["1"].expected_minutes == pytest.approx(one)
    assert p["2"].expected_minutes == pytest.approx(two)
    assert p["3"].expected_minutes == pytest.approx(three)
    assert p["1"].p_start == pytest.approx(1.0)
    assert p["2"].p_start == pytest.approx(1 - alpha)
    assert p["3"].p_start == pytest.approx(alpha)


def test_window_is_the_last_ten_team_games():
    games = [_game("g0", [_row(1, "One", "BOS", 48, started=True)])]
    for i in range(1, 11):
        games.append(_game(f"g{i}", [_row(1, "One", "BOS", 10, started=True)]))

    projections = lineups.project_team("BOS", games)

    assert _by_id(projections)["1"].expected_minutes == pytest.approx(10.0)


def test_player_traded_away_is_not_projected_for_his_old_team():
    games = _three_games()
    projections = lineups.project_team(
        "BOS", games, latest_team={"1": "LAL", "2": "BOS", "3": "BOS"}
    )

    assert "1" not in _by_id(projections)
    assert set(_by_id(projections)) == {"2", "3"}


def test_player_who_joined_mid_window_is_not_diluted_by_games_before_he_arrived():
    games = _three_games()
    games.append(_game("g4", [_row(9, "New", "BOS", 30, started=True)]))

    projections = lineups.project_team("BOS", games)

    assert _by_id(projections)["9"].expected_minutes == pytest.approx(30.0)
    assert _by_id(projections)["9"].p_start == pytest.approx(1.0)


def test_out_player_is_removed_and_his_minutes_go_to_fully_available_teammates():
    games = [
        _game("g1", [
            _row(1, "One", "BOS", 30, started=True),
            _row(2, "Two", "BOS", 20, started=True),
            _row(3, "Three", "BOS", 10),
        ])
    ]
    projections = lineups.project_team("BOS", games, statuses={"1": "Out"})

    p = _by_id(projections)
    assert p["1"].expected_minutes == 0.0
    assert p["1"].p_start == 0.0
    assert p["1"].availability == 0.0
    # 30 removed, split 20:10 between players 2 and 3, who were on 20 and 10.
    assert p["2"].expected_minutes == pytest.approx(40.0)
    assert p["3"].expected_minutes == pytest.approx(20.0)


def test_redistribution_caps_each_player_at_forty_minutes():
    games = [
        _game("g1", [
            _row(1, "One", "BOS", 40, started=True),
            _row(2, "Two", "BOS", 38, started=True),
            _row(3, "Three", "BOS", 2),
        ])
    ]

    p = _by_id(lineups.project_team("BOS", games, statuses={"1": "Out"}))

    assert p["2"].expected_minutes == pytest.approx(40.0)
    assert p["3"].expected_minutes == pytest.approx(40.0)


def test_only_an_out_player_frees_minutes_for_teammates():
    games = [
        _game("g1", [
            _row(1, "One", "BOS", 30, started=True),
            _row(2, "Two", "BOS", 20, started=True),
            _row(3, "Three", "BOS", 10),
        ])
    ]

    p = _by_id(
        lineups.project_team(
            "BOS", games, statuses={"1": "Out", "2": "Questionable"}
        )
    )

    assert p["1"].expected_minutes == 0.0
    assert p["2"].expected_minutes == pytest.approx(10.0)  # scaled, not boosted
    assert p["3"].expected_minutes == pytest.approx(40.0)  # absorbs the Out 30


def test_partial_statuses_scale_minutes_but_do_not_recycle_them_to_each_other():
    games = [
        _game("g1", [
            _row(1, "One", "BOS", 30, started=True),
            _row(2, "Two", "BOS", 20, started=True),
            _row(3, "Three", "BOS", 10),
        ])
    ]
    projections = lineups.project_team(
        "BOS",
        games,
        statuses={"1": "Questionable", "2": "Probable", "3": "Day-To-Day"},
    )

    p = _by_id(projections)
    assert p["1"].expected_minutes == pytest.approx(30 * 0.5)
    assert p["2"].expected_minutes == pytest.approx(20 * 0.85)
    assert p["3"].expected_minutes == pytest.approx(10 * 0.60)
    # Partial statuses are scaled, not redistributed: only Out frees minutes.
    assert p["1"].expected_minutes + p["2"].expected_minutes + p["3"].expected_minutes < 60


def test_unknown_status_leaves_the_projection_alone_and_warns(caplog):
    before = _by_id(lineups.project_team("BOS", _three_games()))["1"]

    with caplog.at_level("WARNING"):
        after = _by_id(
            lineups.project_team("BOS", _three_games(), statuses={"1": "GTD"})
        )["1"]

    assert after.expected_minutes == pytest.approx(before.expected_minutes)
    assert after.p_start == pytest.approx(before.p_start)
    assert "unknown injury status" in caplog.text


@pytest.mark.parametrize(
    "status, expected",
    [
        ("Out", 0.0),
        ("out", 0.0),
        ("Doubtful", 0.10),
        ("Questionable", 0.50),
        ("Probable", 0.85),
        ("Day-To-Day", 0.60),
        (" day to day ", 0.60),
    ],
)
def test_status_probability_table(status, expected):
    assert lineups.status_probability(status) == pytest.approx(expected)


def test_unknown_status_probability_is_none():
    assert lineups.status_probability("Active") is None
    assert lineups.status_probability(None) is None


def test_starting_five_excludes_out_players_and_breaks_ties_on_minutes():
    projections = [
        lineups.PlayerProjection("1", "One", "BOS", "G", 30, 1.0, 0.0),
        lineups.PlayerProjection("2", "Two", "BOS", "G", 30, 0.9, 1.0),
        lineups.PlayerProjection("3", "Three", "BOS", "F", 20, 0.9, 1.0),
        lineups.PlayerProjection("4", "Four", "BOS", "F", 20, 0.8, 1.0),
        lineups.PlayerProjection("5", "Five", "BOS", "C", 20, 0.7, 1.0),
        lineups.PlayerProjection("6", "Six", "BOS", "C", 10, 0.1, 1.0),
    ]

    five = lineups.starting_five(projections)

    # One is out. Two and Three tie on start probability; Two's extra minutes
    # put him first.
    assert [p.player_id for p in five] == ["2", "3", "4", "5", "6"]


def test_with_starter_flags_marks_the_first_five_rows_for_each_team():
    rows = [
        _row(1, "One", "PHI", 34),
        _row(2, "Two", "PHI", 33),
        _row(3, "Three", "PHI", 32),
        _row(4, "Four", "PHI", 31),
        _row(5, "Five", "PHI", 30),
        _row(6, "Six", "PHI", 10),
        _row(7, "Seven", "BOS", 34),
        _row(8, "Eight", "BOS", 33),
    ]

    flagged = lineups.with_starter_flags(rows)

    assert [r["started"] for r in flagged] == [True] * 5 + [False] + [True, True]


def test_load_injury_statuses_reads_gzipped_jsonl_and_keeps_the_latest_date(tmp_path):
    path = tmp_path / "injuries.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"player_id": 1, "status": "Questionable", "date": "2026-01-01"}) + "\n")
        fh.write(json.dumps({"player_id": 1, "status": "Out", "date": "2026-01-02"}) + "\n")
        fh.write(json.dumps({"player_id": 2, "status": "Probable"}) + "\n")
        fh.write("\n")

    statuses = lineups.load_injury_statuses([path])

    assert statuses == {"1": "Out", "2": "Probable"}


def test_no_team_games_projects_nobody():
    assert lineups.project_team("BOS", []) == []
