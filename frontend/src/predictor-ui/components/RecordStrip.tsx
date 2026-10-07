// Synced from predictor-ui@7bccedaf75d4. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { record as fmtRecord } from "../fmt";

/**
 * The record strip: the two numbers the facts carry, and the proportion as
 * geometry.
 *
 * Spec §5a-bis, and this is the component that rule exists for. `record` is
 * `hits: 41, settled: 68`; the facts contain no `60.3`. So the text reads
 * `41/68` and the bar's width carries the proportion. Writing "60.3%" here would
 * be a third number this product invented by arithmetic on two real ones — the
 * same class of error as a model inventing a figure, and the reason the rule is
 * written down rather than left to whoever renders it.
 *
 * When `rebuilt` > 0, the headline includes picks made after tip-off. The
 * pre-tip subset (if available) is shown beside it with the label "Before tip-off".
 * This satisfies the honesty rule: the headline is not labeled "made before tip-off"
 * when it includes post-tip picks.
 */
export function RecordStrip({
  label,
  hits,
  settled,
  rebuilt = 0,
  preTip,
}: {
  label: string;
  hits: number | null;
  settled: number;
  rebuilt?: number;
  preTip?: { hits: number; settled: number } | null;
}) {
  const counts = fmtRecord(hits ?? Number.NaN, settled);
  const has = counts !== "—" && settled > 0;
  const share = has ? Math.max(0, Math.min(1, (hits ?? 0) / settled)) : 0;

  const preTipCounts = preTip && preTip.settled > 0
    ? `${preTip.hits} of ${preTip.settled}`
    : null;

  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <span className="text-xs font-semibold uppercase tracking-wide text-pr-text-dim">{label}</span>
      {has ? (
        <>
          <span
            role="img"
            aria-label={`${hits} of ${settled}`}
            className="flex h-1.5 w-full overflow-hidden rounded-pr bg-pr-panel-2"
          >
            <span data-testid="record-fill" className="h-full bg-pr-accent" style={{ width: `${share * 100}%` }} />
          </span>
          <span className="font-pr-display text-sm font-semibold text-pr-text">{counts}</span>
          {rebuilt > 0 && (
            <span className="text-xs text-pr-text-faint">
              {rebuilt} of them made after tip-off
            </span>
          )}
          {preTipCounts && (
            <>
              <span className="text-xs font-semibold uppercase tracking-wide text-pr-text-dim mt-1.5">Before tip-off</span>
              <span className="font-pr-display text-sm font-semibold text-pr-text">{preTipCounts}</span>
            </>
          )}
        </>
      ) : (
        // Nothing settled reads as a dash, never 0/0 — "0/0" is a claim about a
        // record that does not exist yet.
        <span className="font-pr-display text-sm font-semibold text-pr-text-faint">—</span>
      )}
    </div>
  );
}
