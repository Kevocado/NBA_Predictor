/**
 * NBA's vendored `SignalRows` with one ILLUSTRATIVE absence row.
 *
 * See `frontend/signals-harness.html` for why this exists and why the row is
 * labelled. Nothing on this page is a real NBA observation: every one of the 33 real
 * games on the 2026-10-05 slate answers `{"signals": []}`.
 *
 * The three rows are the three units NBA's `STAT_TARGETS` declares, because "one
 * unit for every stat" was a real bug on this adapter — `UNIT` was
 * `{stat: "pts" for stat in STAT_TARGETS}`, so a rebound projection rendered as
 * "11 pts". Each row states its OWN unit, and a whole number prints as a whole
 * number: `30` is not `30.0`, because a tenth of a point is precision the model never
 * had.
 *
 * All three carry an `n` far below the `n >= 30` rate floor, because for an absence
 * `n` counts INJURED PLAYERS rather than graded games.
 */
import { createRoot } from "react-dom/client";

import { SPEC_MIN_N, SignalRows, type Signal } from "../src/predictor-ui";

const SOURCE = "ESPN injury report, 2026-10-04";

/** `(stat, rank, value, unit)` — one row per declared unit. */
const ROWS: Array<[string, number, number, string]> = [
  ["points", 1, 31.5, "pts"],
  ["rebounds", 1, 11.0, "reb"],
  ["threes", 2, 2.5, "3PM"],
];

/** `points` -> `Points`. The sport's own heading, which is what the adapter's
 *  `stat.capitalize()` already produces -- see `signals/absence.py`. */
const cap = (stat: string) => stat.charAt(0).toUpperCase() + stat.slice(1);

const signals: Signal[] = ROWS.map(([stat, rank, value, unit]) => ({
  kind: "absence",
  sport: "nba",
  game_id: "harness",
  headline: {
    text: `Out: A. Player, our #${rank} ${cap(stat)} projection (${value} ${unit})`,
    figures: { projection: value },
  },
  n: 1,
  source: `${SOURCE} · 1 player out`,
  as_of: "2026-10-04",
  strength: 1 / rank,
  pre_kickoff_only: true,
  visual: "absence_strip",
}));

function Harness() {
  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-sm text-[var(--color-net-faint)]">
        NBA&apos;s absence row — the phase-2 <code>absence_strip</code> visual
      </h1>
      <SignalRows signals={signals} />
      <p className="max-w-[70ch] text-xs text-pr-text-dim">
        Each row states <strong>its own unit</strong>: <code>pts</code>,{" "}
        <code>reb</code>, <code>3PM</code>. The marker is a plain unsigned number
        because this component cannot know the unit — NBA&apos;s figures are points,
        rebounds and threes, and only the adapter knows which it is holding. The
        first version of the adapter labelled them all <code>pts</code>.
      </p>
      <p className="max-w-[70ch] text-xs text-pr-text-dim">
        It is deliberately not a bar (a projection is not a hit rate) and not
        coloured: a player being out is not an outcome the model got right or wrong.
        All three draw at <code>n=1</code>, far below the{" "}
        <code>n ≥ {SPEC_MIN_N}</code> rate floor — for an absence <code>n</code> counts
        injured <em>players</em>, not graded games.
      </p>
    </div>
  );
}

const el = document.getElementById("root");
if (!el) throw new Error("no #root to mount into");
createRoot(el).render(<Harness />);
