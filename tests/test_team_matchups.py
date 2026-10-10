"""NBA team matchup facts: ratings, pace, rebounds, 3P%, FG% as neutral ranked rows."""
import pandas as pd

from nba_predictor.api import facts as facts_mod
from nba_predictor.signals.team_matchups import team_matchups, team_rates
from tests.test_four_factors_duel import DATES, TEAMS, _schedule


def _league(flip: bool = False, with_3pa: bool = True) -> pd.DataFrame:
    """A 30-team league in the pipeline's own frame shape; T29 shoots best (FGM=30+index)."""
    frame = _schedule(DATES).copy()
    if flip:
        for side in ("home", "away"):
            frame[f"{side}_fgm"] = 100 - frame[f"{side}_fgm"]
    if with_3pa:
        for side in ("home", "away"):
            frame[f"{side}_fg3a"] = 25.0
    return frame


def _by_id(rows):
    return {r["id"]: r for r in rows}


def test_rows_have_the_agreed_shape_and_are_neutral():
    rows = team_matchups("T29", "T00", _league(), "2026-03-18")

    assert {r["id"] for r in rows} == {
        "rating:home", "rating:away", "reb:home", "reb:away", "fg3_pct:home", "fg3_pct:away",
        "fg_pct:home", "fg_pct:away", "pace",
    }
    for r in rows:
        assert set(r) == {"id", "attacker", "defender", "stat", "foil", "attacker_rank",
                          "defender_rank", "n_teams", "toward_pick"}
        assert r["toward_pick"] is None
        assert r["n_teams"] == 30
        assert 1 <= r["attacker_rank"] <= 30 and 1 <= r["defender_rank"] <= 30


def test_ranks_come_from_the_games_and_follow_the_league_when_it_flips():
    best = _by_id(team_matchups("T29", "T00", _league(), "2026-03-18"))["fg_pct:home"]
    flipped = _by_id(team_matchups("T29", "T00", _league(flip=True), "2026-03-18"))["fg_pct:home"]

    assert best["attacker"] == "T29" and best["attacker_rank"] == 1
    assert flipped["attacker_rank"] == 30       # the same team, now the worst shooter


def test_defence_is_what_a_team_allows_not_what_it_scores():
    rates = team_rates(_league(), "2026-03-18")
    rows = _by_id(team_matchups("T29", "T00", _league(), "2026-03-18"))

    allowed_rank = int(rates["fg_pct_allowed"].rank(method="min", ascending=True)["T00"])
    assert rows["fg_pct:home"]["defender"] == "T00"
    assert rows["fg_pct:home"]["defender_rank"] == allowed_rank


def test_only_games_before_as_of_are_read():
    assert team_rates(_league(), "2026-03-01").empty


def test_a_league_short_of_thirty_ranked_teams_gives_no_rows():
    league = _league()
    league = league[~league["home_team"].isin(TEAMS[:3]) & ~league["away_team"].isin(TEAMS[:3])]

    assert team_matchups("T29", "T10", league, "2026-03-18") == []


def test_without_three_point_attempts_the_3p_row_is_skipped_not_zeroed():
    ids = {r["id"] for r in team_matchups("T29", "T00", _league(with_3pa=False), "2026-03-18")}

    assert "fg3_pct:home" not in ids and "fg_pct:home" in ids


def test_matchup_rows_reach_the_facts_bundle(monkeypatch):
    monkeypatch.setattr(facts_mod, "_box_score_history", lambda as_of: _league())

    rows = facts_mod._matchup_rows("T29", "T00", "2026-03-18", "home")

    neutral = [r for r in rows if r["id"] in ("pace", "rating:home")]
    assert len(neutral) == 2 and all(r["toward_pick"] is None for r in neutral)
