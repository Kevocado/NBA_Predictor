/**
 * The NBA game's "Model's top calls": four category lists of projections, each
 * row carrying the model's own number, the ± it deserves, and where that ± came
 * from.
 *
 * This is a **presenter**, not a second ranked list. The ranking rules -- the
 * three-row ceiling, the removal of out players, the refusal to draw a
 * probability row as a share bar -- live in `PicksList` (predictor-ui), which is
 * the one implementation every site shares. What this file owns is the part that
 * is specifically NBA's, and that would be a lie to share:
 *
 *  * **There is no probability to show.** `models/player_props.py` is an
 *    XGBRegressor returning a raw point total, and no calibrated probability is
 *    served anywhere in this repo. Every row is therefore `kind: "projection"`
 *    and says "projection, not a probability" on its face, so nobody reads a
 *    27.5 as a 2750% share. Inventing a probability to fill a bar is the one
 *    thing this block must not do.
 *  * **The ± is a per-stat aggregate, not this player's error.** NBA has no
 *    per-player graded ledger: `predict_double_double_probability` exists with
 *    zero callers and nothing grades an individual player's props. So each row
 *    says so in words, and carries only the in-sample MAE for its own stat --
 *    never another stat's, which would be a number borrowed from a different
 *    unit of analysis.
 *  * **A missing MAE is `null`, not zero.** The backend sends `null` when a stat
 *    has no resolved rows. Passing that through as `0` would render "± 0.0",
 *    which claims the model has never missed; `undefined` renders the honest
 *    sentence instead.
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

/**
 * What the row's number is, in words, on the row itself.
 *
 * Not optional context: a reader who lands on "27.5" with no unit has to be told
 * it is a projection and not a chance of something, and the plan requires the
 * row to carry that label rather than the page to imply it once in a header.
 * The heading goes in too, so the label also asserts which of the four lists the
 * figure belongs to.
 */
const rowDetail = (heading: string) => `${heading} projection, not a probability`;

/**
 * The honest provenance sentence for a row whose stat has an error estimate.
 *
 * Two figures now travel with every projection, and the sentence names both
 * rather than quietly picking one: `mae` is the in-sample error over every
 * counted pick for the stat -- one per (game, player, stat), the earliest
 * recorded, whenever it was made -- and `mae_pre_tip` is the same estimate over
 * the picks made before their own tip-off. They differ whenever a model was
 * re-run on a game that had already been played, which is most of the time,
 * and a reader weighing the model against a book has to know which one is on
 * the row.
 */
function provenanceWithMae(
  heading: string,
  preTipMae: number | null | undefined,
  n: number | undefined,
  nPreTip: number | undefined,
): string {
  const scope = `on ${(n ?? 0).toLocaleString("en-US")} resolved ${heading.toLowerCase()} rows across the league`;
  const preTip =
    typeof preTipMae === "number" && Number.isFinite(preTipMae)
      ? ` Of those, the ${(nPreTip ?? 0).toLocaleString("en-US")} made before their own tip-off put it at ±${preTipMae.toFixed(1)}.`
      : " No pick for this stat was made before its game's tip-off, so there is no pre-tip figure beside it.";
  return (
    `Model projection. The ± is the in-sample MAE (mean absolute error) ${scope} -- a per-stat ` +
    `aggregate, not this player's own error, and it counts picks made on a re-run of the model ` +
    `as well as picks made on the night.${preTip} No graded per-player record yet.`
  );
}

/** And for a row whose stat has none. Never "± 0": that is a different claim. */
const provenanceWithoutMae =
  "Model projection. No error estimate yet for this stat -- nothing has been resolved to measure " +
  "against, so there is no ± to show rather than a ±0. No graded per-player record yet.";

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
        detail: rowDetail(heading),
        value: p.predicted_value,
        kind: "projection",
        // `undefined`, not 0, when the API sent null. PicksList turns undefined
        // into "no error estimate yet"; a 0 becomes "± 0.0".
        margin: typeof p.mae === "number" && Number.isFinite(p.mae) ? p.mae : undefined,
        provenance:
          typeof p.mae === "number" && Number.isFinite(p.mae)
            ? provenanceWithMae(heading, p.mae_pre_tip, p.mae_n, p.mae_n_pre_tip)
            : provenanceWithoutMae,
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
