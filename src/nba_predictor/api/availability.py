"""The availability gate, as a pure function over the ESPN injury report.

Every design choice here follows from one asymmetry, measured rather than
assumed: a removal we cannot justify is worse than a removal we miss.

  * A removal we CAN justify is exact. The report and the prop rows both key
    on the ESPN athlete id, so "is this player out?" is a dict lookup. No
    player name is ever compared to another name.
  * Anything we cannot resolve removes NOBODY. Not a near match, not a
    same-named player on another team, not a team-level guess. A false
    removal deletes a real, healthy player from a ranking on the strength of
    a string comparison, and the site's NFL depth-chart code already refuses
    that trade for the same reason -- an unavailable chart leaves is_starter
    as None, because False "would be a different and much worse claim".

So the failure mode here is a missed removal: the player stays in the ranking.
That is visible and correctable. The alternative failure is silent and wrong.
"""

# ESPN's injury report. Provenance for the out line: where a reader can check
# this, which is not the same as ESPN's own "source" field ("basic/manual",
# which describes ESPN's curation rather than the feed's origin).
INJURY_SOURCE = "ESPN injury report"

# Only a plain Out removes a player. Measured on the live report 2026-10-01:
# 52 of 65 entries were "Day-To-Day" and 13 were "Out". Removing Day-To-Day
# too would delete four fifths of every ranking, so it does not.
OUT_STATUSES = frozenset({"out"})

# A doubtful player is not removed and is not silent either. Those 52 entries
# used to reach no feed at all, which reads to a reader as "checked and clear"
# when the truth is "flagged, and we are not telling you" -- and silence is not
# honesty. So they are resolved and surfaced with their status, their feed and
# its date, and left in the ranking.
#
# The exact vocabulary is as narrow as OUT_STATUSES, for the same reason: only
# "Day-To-Day" is the measured word, and widening the match on a guess is how a
# note lands on a player the report never flagged.
DOUBTFUL_STATUSES = frozenset({"day-to-day"})


def is_out(status) -> bool:
    """True only for a status ESPN spells exactly as out -- case aside.

    Strictly a case-insensitive match, nothing looser. "Out (questionable)",
    "Out: rest", "Probably out" and friends must not remove anybody: the
    measured vocabulary is exactly {"Out", "Day-To-Day"}, and widening the
    match on a guess is how a wrong person leaves a ranking.
    """
    return isinstance(status, str) and status.strip().lower() in OUT_STATUSES


def is_doubtful(status) -> bool:
    """True only for a status ESPN spells exactly as day-to-day -- case aside.

    Strictly a case-insensitive match, nothing looser, and never overlapping
    :func:`is_out`: the two feeds are disjoint so that one player is either
    removed or flagged, never both and never neither.

    A doubtful status is NOT a probability and NOT a downgrade coefficient. It
    is the feed's word for "unconfirmed", and this repo has no calibrated
    number to pair it with -- models/player_props.py is an XGBRegressor
    returning a raw point total. Turning it into a number here would invent a
    quantity ESPN does not publish.
    """
    return isinstance(status, str) and status.strip().lower() in DOUBTFUL_STATUSES


def _resolve_players(injuries, player_ids, predicate) -> list[dict]:
    """Entries for the players in ``player_ids`` whose status ``predicate`` accepts.

    The shared exact-key resolution behind both feeds, so the flagging side
    inherits the removal side's asymmetry rather than re-deciding it:
    ``player_ids`` is the set of ids present in the rows about to be ranked, so a
    league-wide report cannot produce entries for players who are not in this
    game's ranking at all. A row whose id is missing or blank resolves to
    nobody. One player with four stat rows yields one entry.

    Every field is the feed's own, verbatim, plus the provenance this repo
    adds: no probability, no coefficient, no restatement of the status.
    """
    wanted = {str(pid) for pid in player_ids or ()}
    resolved: dict[str, dict] = {}

    for injury in injuries or ():
        player_id = str(injury.get("player_id") or "")
        if not player_id or player_id not in wanted:
            continue
        if not predicate(injury.get("status")):
            continue
        if player_id in resolved:
            continue
        resolved[player_id] = {
            "player_id": player_id,
            "player_name": injury.get("player_name") or player_id,
            "team": injury.get("team") or "",
            "status": injury.get("status") or "",
            "source": INJURY_SOURCE,
            "dated": injury.get("dated") or "",
        }

    return list(resolved.values())


def resolve_out_players(injuries, player_ids) -> list[dict]:
    """The players in ``player_ids`` the report says are out.

    A row whose id is missing or blank resolves to nobody. A status that is not
    plainly out resolves to nobody. One player with four stat rows yields one
    entry.
    """
    return _resolve_players(injuries, player_ids, is_out)


def resolve_doubtful_players(injuries, player_ids) -> list[dict]:
    """The players in ``player_ids`` the report says are day-to-day.

    The flag that must not be missing. Same scoping, same exact-id join and
    same one-entry-per-player rule as the out feed; the difference is entirely
    in what the caller does with the result. These players STAY in the ranking
    -- the frontend attaches a note beside the pick and removes nobody.

    Never merged into the out feed on the wire. /players/out is a removal
    instruction that TopCalls.tsx applies to every row it is given, so a
    doubtful player sent there would be deleted by the shipped frontend.
    """
    return _resolve_players(injuries, player_ids, is_doubtful)
