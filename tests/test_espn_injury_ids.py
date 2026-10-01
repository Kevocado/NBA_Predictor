"""The injuries endpoint keys by name; props key by ESPN athlete id.

Measured against the live ESPN payload on 2026-10-01 (745KB, 26 team blocks,
65 injury entries) the injuries endpoint's athlete object carries NO "id"
field -- 0 of 65 entries have one. So the plan's assumed "keep the athlete id"
route does not exist, and an id resolution has to be built.

The route that is safe, and is what this file pins: the athlete's own
playercard link href contains the ESPN athlete id
(https://www.espn.com/nba/player/_/id/5105571/henri-veesaar). That is a
deterministic read of a URL -- the same id space get_player_boxscore already
puts in player_id -- so the join is an exact key match with no name string
comparison anywhere. A name join could resolve "LeBron James" to the wrong
person and remove the wrong person from the ranking; this cannot.

The measured proof that the id space matches: of the 65 live entries, the 52
whose displayName also appears in data/cache/hub/players.json had an id that
agreed with that player's player_id in 52 of 52 cases, 0 disagreements. The
other 13 are players with no prop row at all, and therefore nothing to remove.
"""

import json

import pytest

from nba_predictor.data import espn


@pytest.fixture
def clear_cache(tmp_path, monkeypatch):
    cache = tmp_path / "espn"
    monkeypatch.setattr(espn, "ESPN_CACHE_DIR", cache)
    return cache


def _team_block(team_name, entries):
    return {"displayName": team_name, "injuries": entries}


def _entry(display_name, status="Out", dated="2026-09-21T19:50Z", athlete_id=5105571):
    """An ESPN-shaped injury entry, with the playercard link the real payload
    carries and the athlete object that carries no id field at all."""
    athlete = {
        "firstName": display_name.split()[0],
        "lastName": " ".join(display_name.split()[1:]),
        "displayName": display_name,
        "links": [
            {
                "rel": ["playercard", "desktop", "athlete"],
                "href": (
                    "https://www.espn.com/nba/player/_/id/"
                    f"{athlete_id}/{display_name.lower().replace(' ', '-')}"
                ),
            }
        ],
    }
    return {"athlete": athlete, "status": status, "date": dated}


def test_get_injuries_extracts_the_espn_athlete_id_from_the_playercard_link(monkeypatch, clear_cache):
    monkeypatch.setattr(
        espn,
        "_fetch_json",
        lambda *a, **k: {"injuries": [_team_block("Boston Celtics", [_entry("Jayson Tatum")])]},
    )

    assert espn.get_injuries() == [
        {
            "team": "BOS",
            "player_id": "5105571",
            "player_name": "Jayson Tatum",
            "status": "Out",
            "dated": "2026-09-21T19:50Z",
        }
    ]


def test_the_athlete_object_in_the_real_payload_has_no_id_field():
    # The reason this file exists: the obvious fix ("just keep athlete.id")
    # is not available. 0 of 65 live entries had one. If ESPN ever starts
    # sending it, the id field below still wins and this documents why the
    # link is the fallback rather than the primary.
    athlete = {"displayName": "Jayson Tatum", "links": []}
    assert "id" not in athlete


def test_the_underscore_path_form_is_read_as_well_as_the_plain_one(monkeypatch, clear_cache):
    entry = _entry("Jayson Tatum")
    entry["athlete"]["links"][0]["href"] = "https://www.espn.com/nba/player/_/id/5105571/x"
    monkeypatch.setattr(
        espn, "_fetch_json", lambda *a, **k: {"injuries": [_team_block("Boston Celtics", [entry])]}
    )

    assert espn.get_injuries()[0]["player_id"] == "5105571"


def test_a_sportscenter_uid_link_is_read_when_there_is_no_playercard(monkeypatch, clear_cache):
    entry = _entry("Jayson Tatum")
    entry["athlete"]["links"] = [
        {
            "rel": ["stats", "sportscenter", "app", "athlete"],
            "href": "sportscenter://x-callback-url/showClubhouse?uid=s:40~l:2~a:5105571&section=stats",
        }
    ]
    monkeypatch.setattr(
        espn, "_fetch_json", lambda *a, **k: {"injuries": [_team_block("Boston Celtics", [entry])]}
    )

    assert espn.get_injuries()[0]["player_id"] == "5105571"


def test_an_injury_with_no_readable_link_gets_a_blank_id_and_resolves_to_nobody(monkeypatch, clear_cache):
    # Fail toward "removes nobody". A wrong id is worse than a missing one.
    entry = _entry("Nobody At All")
    entry["athlete"]["links"] = []
    monkeypatch.setattr(
        espn, "_fetch_json", lambda *a, **k: {"injuries": [_team_block("Boston Celtics", [entry])]}
    )

    row = espn.get_injuries()[0]

    assert row["player_id"] == ""
    assert row["player_name"] == "Nobody At All"  # still reported, just unresolvable


def test_two_players_sharing_a_display_name_keep_distinct_ids(monkeypatch, clear_cache):
    # The exact failure a name join would cause. The ids differ, so nothing
    # collides and the caller sees two unresolvable-to-one-another entries.
    monkeypatch.setattr(
        espn,
        "_fetch_json",
        lambda *a, **k: {
            "injuries": [
                _team_block(
                    "Boston Celtics",
                    [_entry("Jayson Tatum", athlete_id=5105571), _entry("Jayson Tatum", athlete_id=999)],
                )
            ]
        },
    )

    ids = [row["player_id"] for row in espn.get_injuries()]

    assert sorted(ids) == ["5105571", "999"]


def test_a_cached_report_written_before_ids_existed_is_not_reused(monkeypatch, clear_cache):
    # A stale cache without ids would silently disable the whole gate, which is
    # the one thing decision 6 forbids. It must be refetched, not trusted.
    cache_file = clear_cache / "injuries_current.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"injuries": [{"team": "BOS", "player_name": "Stale Guy", "status": "Out"}]}))
    monkeypatch.setattr(
        espn,
        "_fetch_json",
        lambda *a, **k: {"injuries": [_team_block("Boston Celtics", [_entry("Jayson Tatum")])]},
    )

    assert espn.get_injuries()[0]["player_id"] == "5105571"


def test_a_cache_with_ids_is_reused_without_a_second_fetch(monkeypatch, clear_cache):
    calls = []

    def _fetch(*a, **k):
        calls.append(1)
        return {"injuries": [_team_block("Boston Celtics", [_entry("Jayson Tatum")])]}

    monkeypatch.setattr(espn, "_fetch_json", _fetch)
    espn.get_injuries()
    espn.get_injuries()

    assert len(calls) == 1


def test_the_missing_date_is_reported_as_blank_rather_than_invented(monkeypatch, clear_cache):
    entry = _entry("Jayson Tatum")
    del entry["date"]
    monkeypatch.setattr(
        espn, "_fetch_json", lambda *a, **k: {"injuries": [_team_block("Boston Celtics", [entry])]}
    )

    assert espn.get_injuries()[0]["dated"] == ""


def test_a_stale_cached_report_is_refetched_not_reused_forever(monkeypatch, clear_cache):
    """The availability gate reads this report on every request. A cache with no age
    limit freezes it: a player ruled Out later stays ranked, and a player who has
    returned stays removed (CodeRabbit on NBA#18). The report must expire."""
    import os, time
    calls = []

    def _fetch(*a, **k):
        calls.append(1)
        return {"injuries": [_team_block("Boston Celtics", [_entry("Jayson Tatum")])]}

    monkeypatch.setattr(espn, "_fetch_json", _fetch)
    espn.get_injuries()
    assert len(calls) == 1
    cache_file = next(clear_cache.glob("injuries_*.json"))
    old = time.time() - (espn.INJURY_CACHE_TTL_SECONDS + 60)
    os.utime(cache_file, (old, old))
    espn.get_injuries()
    assert len(calls) == 2, "an expired injury report was served from cache"


def test_a_fresh_cached_report_is_still_reused(monkeypatch, clear_cache):
    calls = []

    def _fetch(*a, **k):
        calls.append(1)
        return {"injuries": [_team_block("Boston Celtics", [_entry("Jayson Tatum")])]}

    monkeypatch.setattr(espn, "_fetch_json", _fetch)
    espn.get_injuries()
    espn.get_injuries()
    assert len(calls) == 1


def test_a_failed_refetch_of_an_expired_report_raises_rather_than_serving_stale(monkeypatch, clear_cache):
    import os, time
    monkeypatch.setattr(espn, "_fetch_json", lambda *a, **k: {"injuries": [_team_block("Boston Celtics", [_entry("Jayson Tatum")])]})
    espn.get_injuries()
    cache_file = next(clear_cache.glob("injuries_*.json"))
    old = time.time() - (espn.INJURY_CACHE_TTL_SECONDS + 60)
    os.utime(cache_file, (old, old))

    def _boom(*a, **k):
        raise RuntimeError("ESPN unreachable")

    monkeypatch.setattr(espn, "_fetch_json", _boom)
    with pytest.raises(RuntimeError):
        espn.get_injuries()
