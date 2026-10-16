import type { TrackRecordPick } from "../api/client";
import { kickoff, pct } from "../predictor-ui";

const MARK = {
  win: <span className="font-semibold text-pr-win" aria-label="Right">Right</span>,
  loss: <span className="font-semibold text-pr-loss" aria-label="Wrong">Wrong</span>,
  push: <span className="text-pr-text-faint">Push</span>,
};
const markFor = (hit: boolean | null) => (hit === null ? MARK.push : hit ? MARK.win : MARK.loss);

/**
 * Games where the model backed the side the price did not, with how each one ended.
 *
 * NBA-only: the shared kit has no comparison-with-the-odds block, so this stays
 * here. Everything else this file used to hold (the headline, week strip,
 * calibration chart, recent picks, the titled block, the Wilson interval) now
 * lives in the kit at `predictor-ui/src/trackRecord`.
 */
export function Disagreements({
  gameIds,
  picks,
  n,
  hitRate,
}: {
  gameIds: string[];
  picks: TrackRecordPick[];
  n: number;
  hitRate: number | null;
}) {
  const byGame = new Map(picks.filter((p) => p.counted).map((p) => [p.game_id, p]));
  const rows = gameIds.map((id) => byGame.get(id)).filter((p): p is TrackRecordPick => !!p);
  const right = rows.filter((p) => p.hit === true).length;
  return (
    <div>
      <p className="font-pr-display text-2xl font-semibold text-pr-text">
        {n > 0 ? (
          <>
            {hitRate != null ? Math.round(hitRate * n) : right}
            <span className="text-pr-text-faint"> of </span>
            {n}
          </>
        ) : (
          "—"
        )}
        <span className="ml-2 text-sm font-normal text-pr-text-dim">{hitRate != null ? pct(hitRate) : ""}</span>
      </p>
      <p className="text-sm text-pr-text-dim">when the model backed the side the odds did not.</p>
      {rows.length > 0 && (
        <ul className="mt-3 divide-y divide-pr-rule rounded-pr border border-pr-rule bg-pr-panel text-sm">
          {rows.map((p) => (
            <li key={p.game_id} data-testid="disagreement" className="grid grid-cols-[4.5rem_1fr_auto] items-center gap-x-3 px-4 py-2">
              <span className="tabular-nums text-pr-text-faint">{p.gameday ? kickoff(p.gameday) : "—"}</span>
              <span className="truncate"><span className="font-semibold text-pr-text">{p.pick}</span><span className="text-pr-text-faint"> · was {p.actual}</span></span>
              {markFor(p.hit)}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
