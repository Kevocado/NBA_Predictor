"""NBA's `absence` signal.

## Why this sport had to RECOVER the figure rather than read it

PL and NFL both expose a per-player projection on the row that the absence signal
needs. NBA does not: `api/routes.get_game_players` drops an out player's rows
before they are serialised,

    for (player_id, stat), (pick, rebuilt) in picks.items():
        if player_id in out_ids:
            continue

and `OutPlayerOut` carries no `predicted_value`, no stat and no rank — its own
docstring says the player is *"removed, not flagged in place: no list, no bar, no
rank position."*

So there is nothing on the wire to quote. This adapter therefore takes the FULL
`picks_by_player_stat` pool, which is the same dict `get_game_players` iterates
*before* it filters, and ranks from that. The rank stated is the rank the player
HELD, which is the only honest one: filtering first would promote every player
below the absent one, so "#2" would quietly become "#1".

## The figure is a POINTS TOTAL, in the sport's own units

`models/player_props.py` is an XGBRegressor returning a raw number and NBA serves
no calibrated probability anywhere, so there is no share to state and no percentage
is invented. `figures.projection` is the raw `predicted_value`, which is what the
shared component draws as a plain unsigned number.

## Two decisions that are not obvious

**DOUBTFUL IS NOT OUT.** `availability.DOUBTFUL_STATUSES` (`day-to-day`) is a
separate frozenset and those players STAY in the ranking with a note attached. This
adapter reads only the out feed -- partly for the reason
`resolve_doubtful_players`'s docstring gives: `/players/out` is a removal
instruction the shipped frontend applies to every row it is handed.

**TIES ACROSS CATEGORIES GO TO THE PAGE'S OWN DISPLAY ORDER.** NBA's figures are
not comparable across stats: 31.5 points and 2.5 threes are the same number and
different quantities. So a magnitude tiebreak would be a fabricated weighting, and
the tiebreak is instead `STAT_TARGETS`' index -- the order the page lists the
categories in. Ranking by it invents nothing.

**`n` COUNTS INJURED PLAYERS, NOT GRADED GAMES**, which is why no floor is applied
here and why the shared component exempts `absence_strip` from the `n >= 30` rate
floor. Two injured players is not a thin sample of games.
"""
from __future__ import annotations

import math

from ..models.player_props import STAT_TARGETS

#: ESPN's word for out. `availability.OUT_STATUSES` is the source of truth and is
#: read through `is_out` rather than copied here, because a second literal is a
#: second place for the two to disagree -- and the disagreement would be "this row
#: vanished" or "a day-to-day player was deleted".
VISUAL = "absence_strip"
PROJECTION_FIGURE = "projection"

#: The unit in the headline. The component draws a bare number and cannot know it,
#: so the unit lives here, in the adapter that DOES know.
UNIT = {stat: "pts" for stat in STAT_TARGETS}


def _is_out(entry: dict) -> bool:
    """Whether this out-feed entry is genuinely out.

    Reads the feed's own `status` through the same predicate `/players/out` uses,
    so the two can never disagree about who is absent. A doubtful player sent
    through this feed would be REMOVED by the shipped frontend (`TopCalls` applies
    the out feed to every row it is given), so a day-to-day entry here is not a
    cosmetic mistake -- it deletes somebody.
    """
    from ..api.availability import is_out

    return is_out(entry.get("status"))


def _ranked_by_stat(picks: dict) -> dict[str, list[tuple[int, float, str]]]:
    """`{stat: [(rank, predicted_value, player_id), ...]}` over the FULL pool.

    **OUT PLAYERS INCLUDED.** This is the whole point of the module, and the reason
    the adapter takes `picks` rather than the `/players` response: that endpoint
    has already dropped them, so a rank recovered from it would be the rank of a
    list the absent player is not in.

    Rank is 1-based within a stat, descending by `predicted_value` — the same
    ordering and the same one-place convention as `TopCalls.buildTopCalls`, so the
    rank stated here is the rank the page would show if the player were playing.

    Only rows with a FINITE `predicted_value` are ranked. `picks` comes out of a
    store, and a NaN would sort unpredictably and then be stated as "nan pts".
    """
    buckets: dict[str, list[tuple[float, str]]] = {}
    for (player_id, stat), entry in (picks or {}).items():
        pick = (entry or {}).get("pick") or {}
        value = pick.get("predicted_value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        value = float(value)
        if not math.isfinite(value):
            continue
        buckets.setdefault(str(stat), []).append((value, str(player_id)))

    ranked: dict[str, list[tuple[int, float, str]]] = {}
    for stat, rows in buckets.items():
        rows.sort(key=lambda item: item[0], reverse=True)
        ranked[stat] = [(i + 1, value, pid) for i, (value, pid) in enumerate(rows)]
    return ranked


def headline(name: str, stat: str, rank: int, value: float) -> str:
    """The one line, at most 12 words, stating the figure the marker draws.

    The value is formatted by the SAME rule the shared component uses to draw it:
    a whole number prints as a whole number, a fractional one keeps one decimal.
    A 30.0-point projection drawn beside "30 pts" would be the marker and the words
    disagreeing about their own precision on one row, and
    `SignalRows.assertFigureIsStated` refuses a mismatch rather than rendering it.

    The category is the stat's own word, CAPITALISED, and not a heading table
    duplicated here: `STAT_TARGETS` is `points, rebounds, assists, threes`, so
    `capitalize()` yields "Points"/"Rebounds"/"Assists"/"Threes" -- exactly the
    headings `TopCalls.CATEGORIES` maps the same four keys to. A second table would
    be a second place for a new stat target to be added to one and not the other,
    and the failure would be a headline reading "our #1 points projection" beside a
    list headed "Points". The figure means nothing without its category anyway: 31.5
    is not interpretable on its own.

    `{value:g}` is the same rule the shared component draws with -- a whole number
    prints as a whole number, a fractional one keeps one decimal -- so the words and
    the marker cannot disagree about precision.
    """
    shown = f"{value:g}"
    unit = UNIT.get(stat, "")
    tail = f" {unit}" if unit else ""
    return f"Out: {name}, our #{rank} {stat.capitalize()} projection ({shown}{tail})"


def absence_signal(
    game_id,
    picks: dict,
    out_players: list[dict],
    name_by_id: dict | None = None,
) -> dict | None:
    """The single `absence` signal this game can honestly carry, or None.

    ONE row: the out player who held the best rank in their own category. A game
    has many injured players and the endpoint keeps 2-3 signals in total, so "the
    absence that cost the most" is the claim a reader can act on. Every injured
    player is a squad note.

    Returns None -- rather than a row with nothing to draw -- when:
      * the out feed is empty, or every entry in it is doubtful rather than out;
      * the pool is empty, or no out player appears in it (an injured player with
        no projection has no figure to draw and no rank to state);
      * every out player's rows lack a finite `predicted_value`.

    The first of those is the common case, not an error: most games have nobody
    out, and spec §2 is explicit that a fixture with nothing to say renders no rows.
    """
    candidates = []
    for entry in out_players or []:
        if not isinstance(entry, dict) or not _is_out(entry):
            continue
        player_id = str(entry.get("player_id") or "")
        if not player_id:
            continue
        for stat, rows in _ranked_by_stat(picks).items():
            for rank, value, pid in rows:
                if pid != player_id:
                    continue
                candidates.append((rank, stat, value, player_id, entry))

    if not candidates:
        return None

    out_total = sum(
        1
        for entry in out_players or []
        if isinstance(entry, dict) and _is_out(entry) and str(entry.get("player_id") or "")
    )
    if out_total <= 0:
        return None

    # Best rank wins. A tie goes to the category the page lists FIRST, never to
    # the bigger number: 31.5 points and 2.5 threes are the same number and
    # different quantities, so ranking them against each other would be a
    # weighting this repo has no basis for. `STAT_TARGETS`' order is the page's own
    # display order, so following it states nothing.
    candidates.sort(key=lambda c: (c[0], _category_order(c[1])))
    rank, stat, value, player_id, entry = candidates[0]

    names = name_by_id or {}
    name = names.get(player_id) or player_id

    return {
        "kind": "absence",
        "sport": "nba",
        "game_id": str(game_id),
        "headline": {
            "text": headline(name, stat, rank, value),
            "figures": {PROJECTION_FIGURE: value},
        },
        # Injured PLAYERS, not graded games. See the module docstring.
        "n": out_total,
        # ESPN's own provenance, carried through rather than restated as "injured",
        # which would be this repo's word for a status the feed publishes.
        "source": f"{entry.get('source') or 'ESPN injury report'}, {entry.get('dated') or 'undated'}",
        "as_of": entry.get("dated") or "",
        # Rises as the rank improves, because the rank IS the claim this row makes
        # and the endpoint orders rows by it. Bounded and monotone, so two rows can
        # never tie at an arbitrary ceiling. Not derived from the figure: NBA's
        # figures are not comparable across stats.
        "strength": 1.0 / rank,
        "pre_kickoff_only": True,
        "visual": VISUAL,
    }


def _category_order(stat: str) -> int:
    """`stat`'s index in `STAT_TARGETS`, or last for one this repo has not declared.

    Unknown categories sort last rather than raising: a new stat target should
    still produce a row, not take the endpoint down.
    """
    try:
        return STAT_TARGETS.index(stat)
    except ValueError:
        return len(STAT_TARGETS)
