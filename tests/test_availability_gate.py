"""The availability gate: an out player leaves the ranking entirely.

The measured asymmetry, stated once and asserted on both sides:

  A removal we can prove (the injury carries the same ESPN athlete id as the
  prop row) is exact and safe.

  Anything we cannot resolve removes NOBODY. Not a near match, not a
  same-name player on another team, not a team-only guess. A false removal
  deletes a real, healthy player from a ranking on the strength of a string
  comparison, and the site's own NFL depth-chart code already refuses that
  trade for the same reason: an unavailable chart leaves is_starter as None
  because False "would be a different and much worse claim".
"""

import pytest

from nba_predictor.api import availability


def _injury(player_id, name, status="Out", team="BOS", dated="2026-09-21T19:50Z"):
    return {
        "team": team,
        "player_id": player_id,
        "player_name": name,
        "status": status,
        "dated": dated,
    }


def test_only_the_out_status_removes_a_player():
    assert availability.is_out("Out") is True


def test_day_to_day_does_not_remove_a_player():
    # 52 of the 65 live ESPN entries are Day-To-Day. Removing them would
    # delete four fifths of every ranking, so "Out" is the only remover.
    assert availability.is_out("Day-To-Day") is False


@pytest.mark.parametrize(
    "status",
    [
        "",
        None,
        "Questionable",
        "Probable",
        "Active",
        "Day-To-Day",
        # Looser than the measured vocabulary. Nothing widens the match on a
        # guess: only "Out" removes.
        "Out (questionable)",
        "Out: rest",
        "Probably out",
        "Not out",
    ],
)
def test_anything_that_is_not_plainly_out_does_not_remove(status):
    assert availability.is_out(status) is False


@pytest.mark.parametrize("status", ["Out", "out", "OUT", " out "])
def test_the_out_status_is_matched_case_insensitively(status):
    # ESPN spells it "Out"; case alone must never be the reason a real
    # absence is reported as present.
    assert availability.is_out(status) is True


def test_a_matching_out_injury_removes_that_player():
    injuries = [_injury("5105571", "Jayson Tatum")]

    out = availability.resolve_out_players(injuries, {"5105571", "3934672"})

    assert [entry["player_id"] for entry in out] == ["5105571"]
    assert out[0]["player_name"] == "Jayson Tatum"
    assert out[0]["status"] == "Out"
    assert out[0]["source"] == availability.INJURY_SOURCE
    assert out[0]["dated"] == "2026-09-21T19:50Z"


def test_an_out_player_with_no_prop_row_is_not_an_out_entry():
    # The injury feed is league-wide; a game's rows are a subset.
    injuries = [_injury("5105571", "Jayson Tatum")]

    assert availability.resolve_out_players(injuries, {"3934672"}) == []


def test_an_injury_whose_id_is_blank_removes_nobody():
    assert availability.resolve_out_players([_injury("", "Jayson Tatum")], {"5105571", ""}) == []


def test_an_injury_missing_the_id_field_entirely_removes_nobody():
    # A report shaped by an older cache or a different feed: unresolvable, so
    # it must not remove anyone rather than fall back to the name.
    assert availability.resolve_out_players([{"player_name": "Jayson Tatum", "status": "Out"}], {"5105571"}) == []


def test_an_injury_named_like_a_prop_row_but_with_another_id_removes_nobody():
    # The failure a name join would cause: the name matches, the id does not,
    # and the person with that name who IS in the ranking stays in it.
    injuries = [_injury("5105571", "Jayson Tatum")]

    out = availability.resolve_out_players(injuries, {"9999999"})

    assert out == []


def test_a_repeated_injury_for_one_player_yields_one_out_entry():
    injuries = [_injury("5105571", "Jayson Tatum"), _injury("5105571", "J. Tatum")]

    assert len(availability.resolve_out_players(injuries, {"5105571"})) == 1


def test_a_day_to_day_player_stays_in_the_ranking_entirely():
    # No out entry either: flagging in place is forbidden by the spec, and the
    # row is simply not a removal.
    injuries = [_injury("5105571", "Jayson Tatum", status="Day-To-Day")]

    assert availability.resolve_out_players(injuries, {"5105571"}) == []


def test_the_source_names_the_feed_rather_than_espns_internal_label():
    # ESPN's entry carries source="basic/manual", which describes ESPN's own
    # curation, not where a reader can check us. Provenance is the feed.
    assert "ESPN" in availability.INJURY_SOURCE


def test_a_missing_date_is_reported_as_blank_not_as_today():
    out = availability.resolve_out_players([_injury("5105571", "Jayson Tatum", dated="")], {"5105571"})

    assert out[0]["dated"] == ""


def test_no_injuries_removes_nobody():
    assert availability.resolve_out_players([], {"5105571"}) == []
