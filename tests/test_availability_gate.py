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


# ---------------------------------------------------------------------------
# DOUBTFUL: flagged in place, never removed.
#
# The defect this closes, measured on the live report 2026-10-01: of 65 ESPN
# entries, 52 were "Day-To-Day" and 13 were "Out". Only a plain Out removes, so
# those 52 players were ranked and flagged by NOTHING -- which reads to a reader
# as "checked and clear" when the truth is "flagged, and we are not telling
# you". Silence is not honesty. So a doubtful player is RESOLVED, carries its
# status with provenance, and stays in the ranking.
#
# Removal is not on the table for a doubtful status. Four fifths of the report
# is doubtful; removing it deletes four fifths of every ranking on the strength
# of a word ESPN reserves for "unconfirmed", which is the exact false-removal
# trade the gate above exists to refuse.
# ---------------------------------------------------------------------------


def test_day_to_day_is_doubtful():
    assert availability.is_doubtful("Day-To-Day") is True


@pytest.mark.parametrize("status", ["Day-to-day", "DAY-TO-DAY", " day-to-day "])
def test_the_doubtful_status_is_matched_case_insensitively(status):
    # ESPN spells it "Day-To-Day"; case alone must never be the reason a real
    # flag is dropped, exactly as for Out.
    assert availability.is_doubtful(status) is True


@pytest.mark.parametrize(
    "status",
    [
        "",
        None,
        "Out",
        "Questionable",
        "Probable",
        "Active",
        # Looser than the measured vocabulary. Only "Day-To-Day" is doubtful:
        # widening the match on a guess would put a note on a pick the feed
        # never flagged.
        "Day To Day",
        "Day-to-day (out)",
        "Probably day-to-day",
    ],
)
def test_anything_that_is_not_plainly_day_to_day_is_not_doubtful(status):
    assert availability.is_doubtful(status) is False


def test_the_two_statuses_are_disjoint():
    # The feeds must never overlap, or a player is removed and flagged at once.
    for status in ("Out", "Day-To-Day", "Questionable", "", None):
        assert not (availability.is_out(status) and availability.is_doubtful(status))


def test_a_day_to_day_injury_yields_a_doubtful_entry_with_its_source_and_date():
    injuries = [_injury("5105571", "Jayson Tatum", status="Day-To-Day")]

    doubtful = availability.resolve_doubtful_players(injuries, {"5105571", "3934672"})

    assert [entry["player_id"] for entry in doubtful] == ["5105571"]
    assert doubtful[0]["player_name"] == "Jayson Tatum"
    assert doubtful[0]["team"] == "BOS"
    # The feed's own word, verbatim. Not "questionable", not "may not play" --
    # a rewording is an interpretation the report does not make.
    assert doubtful[0]["status"] == "Day-To-Day"
    assert "ESPN" in doubtful[0]["source"]
    assert doubtful[0]["dated"] == "2026-09-21T19:50Z"


def test_a_day_to_day_player_is_still_not_an_out_entry():
    # The headline: resolved as doubtful, and still ranked. Removal is reserved
    # for a plain Out and is never reachable from a doubtful status.
    injuries = [_injury("5105571", "Jayson Tatum", status="Day-To-Day")]

    assert availability.resolve_out_players(injuries, {"5105571"}) == []
    assert len(availability.resolve_doubtful_players(injuries, {"5105571"})) == 1


def test_a_doubtful_player_with_no_prop_row_is_not_a_doubtful_entry():
    # The report is league-wide; a game's rows are a subset. Same scoping as out.
    injuries = [_injury("5105571", "Jayson Tatum", status="Day-To-Day")]

    assert availability.resolve_doubtful_players(injuries, {"3934672"}) == []


def test_a_doubtful_injury_with_a_blank_id_resolves_to_nobody():
    injuries = [_injury("", "Jayson Tatum", status="Day-To-Day")]

    assert availability.resolve_doubtful_players(injuries, {"5105571", ""}) == []


def test_a_doubtful_injury_missing_the_id_field_resolves_to_nobody():
    assert availability.resolve_doubtful_players(
        [{"player_name": "Jayson Tatum", "status": "Day-To-Day"}], {"5105571"}
    ) == []


def test_a_doubtful_injury_named_like_a_prop_row_but_with_another_id_resolves_to_nobody():
    # The failure a name join would cause on the flagging side as much as the
    # removal side: the name matches, the id does not.
    injuries = [_injury("5105571", "Jayson Tatum", status="Day-To-Day")]

    assert availability.resolve_doubtful_players(injuries, {"9999999"}) == []


def test_a_repeated_doubtful_injury_for_one_player_yields_one_entry():
    injuries = [
        _injury("5105571", "Jayson Tatum", status="Day-To-Day"),
        _injury("5105571", "J. Tatum", status="Day-To-Day"),
    ]

    assert len(availability.resolve_doubtful_players(injuries, {"5105571"})) == 1


def test_an_out_player_is_never_a_doubtful_entry():
    # The reverse disjointness: one player, one feed. Otherwise a reader would
    # see a player both removed and merely flagged.
    injuries = [_injury("5105571", "Jayson Tatum", status="Out")]

    assert availability.resolve_doubtful_players(injuries, {"5105571"}) == []


def test_a_doubtful_entry_carries_no_probability_and_no_coefficient():
    # A status is not a number. Upgrading it to a probability or a downgrade
    # coefficient would be inventing a quantity ESPN does not publish, and the
    # repo's own rule is that no probability ships without calibration
    # evidence. The entry is exactly the provenance fields and nothing else.
    injuries = [_injury("5105571", "Jayson Tatum", status="Day-To-Day")]

    entry = availability.resolve_doubtful_players(injuries, {"5105571"})[0]

    assert set(entry) == {"player_id", "player_name", "team", "status", "source", "dated"}
    assert not any(
        token in field.lower() for field in entry for token in ("prob", "pct", "percent", "chance", "odds", "coef")
    )
    for value in entry.values():
        assert isinstance(value, str), f"{value!r} is not the verbatim string the feed carried"


def test_a_doubtful_entry_reports_a_blank_date_as_blank_not_as_today():
    doubtful = availability.resolve_doubtful_players(
        [_injury("5105571", "Jayson Tatum", status="Day-To-Day", dated="")], {"5105571"}
    )

    assert doubtful[0]["dated"] == ""


def test_no_injuries_produces_no_doubtful_entries():
    assert availability.resolve_doubtful_players([], {"5105571"}) == []


def test_the_live_shaped_report_resolves_52_doubtful_and_13_out():
    """The measured split, from a fixture shaped like the live feed.

    Built rather than hand-picked: 65 entries, the 52/13 split measured on the
    real report 2026-10-01, every one of them ranked. This is the count the
    change is FOR -- before it, the 52 produced no entry on any feed and were
    therefore invisible.
    """
    injuries = [
        _injury(str(1000 + i), f"Player {i}", status="Day-To-Day" if i < 52 else "Out")
        for i in range(65)
    ]
    ranked = {str(1000 + i) for i in range(65)}

    doubtful = availability.resolve_doubtful_players(injuries, ranked)
    out = availability.resolve_out_players(injuries, ranked)

    assert len(injuries) == 65
    assert len(doubtful) == 52
    assert len(out) == 13
    # Every one of the 65 resolves exactly once, into exactly one of the two
    # feeds -- so the union is the whole report and the intersection is empty.
    assert len({e["player_id"] for e in doubtful} | {e["player_id"] for e in out}) == 65
    assert {e["player_id"] for e in doubtful} & {e["player_id"] for e in out} == set()
