"""The 5% edge gate: which stored market rows are worth a reader's attention.

Spec section 8, gap G5. Edges were computed and stored and rendered, and
nothing ever selected one -- so a 9% edge and a 1% edge were the same kind of
object. This is the gate that tells them apart.

The sibling contract (NFL `build_value_bet_table` at `edge_threshold=0.05`, PL's
5% x data-confidence) plus three rules this repo needs on its own:

* **One single per game, highest edge wins.** Two flagged singles on one event is
  two opinions where only one can be right.
* **Odds no older than an hour, and a stale slate yields ZERO picks.** Not a
  filtered subset. A stale row that still clears the bar is the dangerous case:
  it looks actionable and is not (spec section 14.3). The refresh loop runs
  continuously, so staleness has to be checked at selection time, not assumed
  away upstream.
* **Same book, same point, both sides.** An edge between one book's total and
  another's is a difference between two shops' prices, not a disagreement with
  the market. This is the NFL post-review fix, and it applies to any two-sided
  market, not just totals.

No line, no cover probability, therefore no edge -- CFB's bug was inventing a
0.5 spread to cover against, and inventing a line here is the same fabrication
one level up. Parlays are excluded outright: their legs are correlated and this
repo does not model that, so a parlay edge is not an edge.

This module selects rows. It places nothing, publishes nothing, posts nothing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

#: Minimum absolute edge, per the sibling contract.
EDGE_THRESHOLD = 0.05

#: A line older than this is not a bettable price.
MAX_ODDS_AGE = timedelta(hours=1)

#: Markets with two sides that must agree on book and point before they can be
#: compared at all. `h2h` has no point.
TWO_SIDED_MARKETS = ("spread", "total")

SIDES = {
    "total": ("over", "under"),
    "spread": ("home", "away"),
}


def _parse_ts(value) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _is_fresh(row: dict, now: datetime) -> bool:
    """False when the stamp is unreadable.

    An unparseable timestamp is treated as stale rather than fresh. Defaulting
    it to "now" would make the oldest possible line the freshest.
    """
    stamp = _parse_ts(row.get("created_at"))
    if stamp is None:
        return False
    return (now - stamp) <= MAX_ODDS_AGE


def pair_total_sides(rows: list[dict]):
    """The (over, under) pair for one book at one point, or None.

    None when the two sides are not comparable: different books, different
    points, or a side missing. Comparing across books measures the difference
    between two shops' prices; comparing across points compares a total to a
    different total.
    """
    sides = SIDES["total"]
    found = {r.get("selection"): r for r in rows if r.get("market") == "total"}
    over, under = found.get(sides[0]), found.get(sides[1])
    if over is None or under is None:
        return None
    if over.get("bookmaker") != under.get("bookmaker"):
        return None
    if over.get("point") is None or over.get("point") != under.get("point"):
        return None
    return over, under


def _has_counterpart(row: dict, rows: list[dict]) -> bool:
    """A two-sided market row needs its other side: same GAME, same book, same point.

    Scoped to the row's own game because `/value-picks` hands the gate every
    recent row at once. Without the game filter, a total for one game could be
    "paired" with a total from another, or with another book's -- which would
    admit a cross-game edge, or reject a perfectly good row because some other
    game happened to have an unmatched side.
    """
    market = row.get("market")
    if market not in TWO_SIDED_MARKETS:
        return True

    same_game = [
        r for r in rows
        if r.get("game_id") == row.get("game_id") and r.get("market") == market
    ]
    if market == "total":
        pair = pair_total_sides(same_game)
        return pair is not None and any(r is row or r == row for r in pair)

    home, away = SIDES["spread"]
    found = {r.get("selection"): r for r in same_game}
    other = found.get(away if row.get("selection") == home else home)
    return (
        other is not None
        and other.get("bookmaker") == row.get("bookmaker")
        and other.get("point") == row.get("point")
    )


def gated_picks(
    rows: list[dict],
    now: datetime | None = None,
    threshold: float = EDGE_THRESHOLD,
) -> list[dict]:
    """The rows worth flagging: at most one single per game, highest edge first.

    `rows` are `game_market_predictions` rows. Every one of these rules is a
    refusal, so the safe failure is an empty list -- never a flagged row we
    cannot justify.
    """
    now = now or datetime.now(timezone.utc)
    if not rows:
        return []

    eligible: list[dict] = []
    for row in rows:
        if row.get("market") == "parlay":
            continue  # correlated legs, unmodelled here

        point = row.get("point")
        if row.get("market") in TWO_SIDED_MARKETS and point is None:
            continue  # no line, no cover probability, no edge

        edge = row.get("edge")
        if edge is None or edge < threshold:
            continue

        if not _is_fresh(row, now):
            continue

        if not _has_counterpart(row, rows):
            continue

        eligible.append(row)

    if not eligible:
        return []

    # One single per game: the highest edge wins, ties broken by the earlier
    # snapshot so the choice does not depend on row order.
    best: dict[str, dict] = {}
    for row in eligible:
        current = best.get(row["game_id"])
        if current is None or (row["edge"], current["created_at"]) > (
            current["edge"],
            row["created_at"],
        ):
            best[row["game_id"]] = row

    return sorted(best.values(), key=lambda r: -r["edge"])