/**
 * The NBA game's "Model's top calls": four category lists of projections, each
 * row carrying the player and the model's own number for that number's category.
 *
 * That is the whole row. Kevin, 2026-10-01: a top call is simple -- the player,
 * the team and the prediction. So no row carries a provenance sentence, a ±
 * margin, a "no graded record" line, a calibration note or availability text,
 * and a stat with no error estimate is not given a sentence saying so; it just
 * shows its figure.
 *
 * This is a **presenter**, not a second ranked list. The ranking rules -- the
 * three-row ceiling, the removal of out players, the refusal to draw a
 * probability row as a share bar -- live in `PicksList` (predictor-ui), which is
 * the one implementation every site shares. What this file owns is the part that
 * is specifically NBA's, and that would be a lie to share:
 *
 *  * **There is no probability to show.** `models/player_props.py` is an
 *    XGBRegressor returning a raw point total, and no calibrated probability is
 *    served anywhere in this repo. Every row is therefore `kind: "projection"`,
 *    which is what stops PicksList drawing a 27.5 as a 2750% share. Inventing a
 *    probability to fill a bar is the one thing this block must not do.
 *  * **`mae` is not carried.** The API sends a per-stat in-sample MAE and NBA has
 *    no per-player graded ledger, so the figure was never this player's own error.
 *    It is not rendered anywhere now and the row does not pass it: a ± nobody
 *    asked for, next to a number it does not belong to, is the kind of thing the
 *    simplification is for.
 *  * **The category is the list's heading, not a sentence on the row.** `detail`
 *    is the bare category name, so the row says which stat it is and nothing
 *    about how confident anyone is about it.
 */
import { PicksList, MAX_ROWS_PER_CATEGORY, type OutPlayer as RowOutPlayer, type PickRow } from "../predictor-ui";
import type { OutPlayer, PlayerProp } from "../api/client";

/** The categories this feed carries, in the order a reader wants them. The
 *  backend's `STAT_TARGETS`; named here because a category heading is words,
 *  not a key. */
const CATEGORIES = [
  { stat: "points", heading: "Points" },
  { stat: "rebounds", heading: "Rebounds" },
  { stat: "assists", heading: "Assists" },
  { stat: "threes", heading: "Threes" },
] as const;

export interface TopCallsData {
  categories: { category: string; rows: PickRow[] }[];
  out: RowOutPlayer[];
}

/**
 * The rows for one category, ranked and capped.
 *
 * A category nobody projected anything for is **absent**, not present-and-empty:
 * an empty list under a heading reads as "the model had nothing to say here",
 * which is a claim about the model rather than about the feed. The cap is the
 * shared component's; this only orders and slices to the same number, so the
 * ranking a reader sees is decided in one place.
 */
export function buildTopCalls(props: PlayerProp[], out: OutPlayer[]): TopCallsData {
  // Removal first, and by the out feed alone.
  //
  // The API already withholds an out player's rows, so re-checking this is
  // redundant *while the API is right*. It is here because the rule is a page
  // rule and the redundancy is the enforcement: a props feed that still carried
  // him -- a stale cache, a partially-applied deploy -- would otherwise put a
  // player nobody can bet on back into the ranking, and PicksList can only refuse
  // a row it is *given* as out, which this one no longer is.
  const outIds = new Set(out.map((p) => p.player_id));

  const categories = CATEGORIES.map(({ stat, heading }) => {
    const rows = props
      .filter((p) => p.stat === stat && !outIds.has(p.player_id))
      .sort((a, b) => b.predicted_value - a.predicted_value)
      .slice(0, MAX_ROWS_PER_CATEGORY)
      .map<PickRow>((p) => ({
        key: `${p.player_id}-${p.stat}`,
        name: p.player_name,
        // No team on this feed: it is joined from the season hub feed by the box
        // score. Left off rather than guessed -- PicksList renders `team`
        // optionally, and a wrong abbreviation next to a real name is worse
        // than none.
        detail: heading,
        value: p.predicted_value,
        kind: "projection",
      }));
    return { category: heading, rows };
  }).filter((c) => c.rows.length > 0);

  // One mention each, whatever their number of stat rows: the feed is per
  // player, but de-duplicating here means a feed that repeats a player cannot
  // print him twice in the one place he is allowed to appear.
  const seen = new Set<string>();
  const outRows = out
    .filter((p) => (seen.has(p.player_id) ? false : (seen.add(p.player_id), true)))
    .map((p) => ({ name: p.player_name, team: p.team, source: p.source, dated: p.dated }));

  return { categories, out: outRows };
}

export function TopCalls({ props, out }: { props: PlayerProp[]; out: OutPlayer[] }) {
  const { categories, out: outRows } = buildTopCalls(props, out);
  return <PicksList title="Model's top calls" categories={categories} out={outRows} />;
}

export default TopCalls;
