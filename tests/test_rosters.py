"""Current rosters from ESPN: who is on which team NOW, with injury status."""
import pytest

from nba_predictor.data import rosters

TEAMS = {"sports": [{"leagues": [{"teams": [
    {"team": {"id": "20", "displayName": "Philadelphia 76ers", "abbreviation": "PHI"}},
    {"team": {"id": "2", "displayName": "Boston Celtics", "abbreviation": "BOS"}},
]}]}]}

ROSTER = {
    "20": {"athletes": [
        {"id": "3917376", "displayName": "Jaylen Brown", "position": {"abbreviation": "SG"}, "injuries": []},
        {"id": "1966", "displayName": "LeBron James", "position": {"abbreviation": "SF"},
         "injuries": [{"status": "Day-To-Day", "date": "2026-10-19T10:00Z"}, {"status": "Out", "date": "2026-10-20T10:00Z"}]},
    ]},
    "2": {"athletes": [{"id": "4251", "displayName": "Paul George", "position": {"abbreviation": "SF"}}]},
}


def _fetch(url, params=None):
    if url.endswith("/teams"):
        return TEAMS
    return ROSTER[url.split("/teams/")[1].split("/")[0]]


def test_each_player_is_on_exactly_the_team_espn_lists_now(monkeypatch):
    monkeypatch.setattr(rosters.espn, "_fetch_json", _fetch)
    by_team = rosters.fetch_rosters()
    assert sorted(by_team) == ["BOS", "PHI"]
    assert {p["player_id"] for p in by_team["PHI"]} == {"3917376", "1966"}
    assert {p["player_id"] for p in by_team["BOS"]} == {"4251"}
    brown = next(p for p in by_team["PHI"] if p["player_id"] == "3917376")
    assert (brown["player_name"], brown["position"], brown["injury_status"]) == ("Jaylen Brown", "SG", "")


def test_the_latest_injury_entry_wins(monkeypatch):
    monkeypatch.setattr(rosters.espn, "_fetch_json", _fetch)
    lebron = next(p for p in rosters.fetch_rosters()["PHI"] if p["player_id"] == "1966")
    assert lebron["injury_status"] == "Out"


def test_a_player_appears_on_one_team_only(monkeypatch):
    dup = {**ROSTER, "2": {"athletes": ROSTER["20"]["athletes"][:1]}}   # ESPN lists Brown on both for a moment
    monkeypatch.setattr(rosters.espn, "_fetch_json", lambda url, params=None: TEAMS if url.endswith("/teams") else dup[url.split("/teams/")[1].split("/")[0]])
    by_team = rosters.fetch_rosters()
    owners = [t for t, ps in by_team.items() if any(p["player_id"] == "3917376" for p in ps)]
    assert len(owners) == 1


def test_a_failed_fetch_serves_the_last_good_rosters_and_never_invents_any(monkeypatch, tmp_path):
    monkeypatch.setattr(rosters.espn, "ESPN_CACHE_DIR", tmp_path)
    monkeypatch.setattr(rosters.espn, "_fetch_json", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    assert rosters.get_rosters() is None                                   # nothing cached, feed down: unknown, not empty
    rosters.espn._save_cache("rosters", "current", {"teams": {"PHI": [{"player_id": "1", "player_name": "X", "position": "G", "injury_status": ""}]}})
    assert rosters.get_rosters(max_age_seconds=0) == {"PHI": [{"player_id": "1", "player_name": "X", "position": "G", "injury_status": ""}]}
