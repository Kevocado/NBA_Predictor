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
# too would delete four fifths of every ranking, so it does not. An
# unavailable-but-unconfirmed player is left in and flagged by nothing, which
# is the conservative direction.
OUT_STATUSES = frozenset({"out"})


def is_out(status) -> bool:
    """True only for a status ESPN spells exactly as out -- case aside.

    Strictly a case-insensitive match, nothing looser. "Out (questionable)",
    "Out: rest", "Probably out" and friends must not remove anybody: the
    measured vocabulary is exactly {"Out", "Day-To-Day"}, and widening the
    match on a guess is how a wrong person leaves a ranking.
    """
    return isinstance(status, str) and status.strip().lower() in OUT_STATUSES


def resolve_out_players(injuries, player_ids) -> list[dict]:
    """The players in ``player_ids`` the report says are out.

    ``player_ids`` is the set of ids present in the rows about to be ranked, so
    a league-wide report cannot produce out entries for players who are not in
    this game's ranking at all.

    A row whose id is missing or blank resolves to nobody. A status that is not
    plainly out resolves to nobody. One player with four stat rows yields one
    entry.
    """
    wanted = {str(pid) for pid in player_ids or ()}
    resolved: dict[str, dict] = {}

    for injury in injuries or ():
        player_id = str(injury.get("player_id") or "")
        if not player_id or player_id not in wanted:
            continue
        if not is_out(injury.get("status")):
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
