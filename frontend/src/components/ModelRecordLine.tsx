import type { TrackRecord } from "../api/client";
import { toRecordModel } from "../lib/recordModel";

export type PreTipRecord = { hits: number; settled: number };

/**
 * The one number the games surfaces may quote: the game_outcome row's pre-tip
 * tally. The headline (all recorded picks, including reruns after tip-off) is
 * the record page's business, not a games page's.
 */
export function preTipRecord(rows: TrackRecord[] | null): PreTipRecord | null {
  if (!rows) return null;
  const market = toRecordModel(rows).markets.find((m) => m.key === "game_outcome");
  const preTip = market?.preKickoff;
  if (!preTip || preTip.n <= 0) return null;
  return { hits: preTip.correct, settled: preTip.n };
}

/**
 * "Model, picks made before tip-off: 33 of 66 · Track record", or nothing.
 * Silent while loading and after a failed read: a wrong or a 0/0 record is
 * worse than no line. `format="percent"` is the modal's tighter reading.
 */
export function ModelRecordLine({
  rows,
  format = "count",
  className = "",
}: {
  rows: TrackRecord[] | null;
  format?: "count" | "percent";
  className?: string;
}) {
  const record = preTipRecord(rows);
  if (!record) return null;
  const figure =
    format === "percent"
      ? `${Math.round((record.hits / record.settled) * 100)}%`
      : `${record.hits} of ${record.settled}`;
  return (
    <p className={`text-xs text-pr-text-dim ${className}`.trim()}>
      Model, picks made before tip-off: <span className="font-semibold text-pr-text">{figure}</span> ·{" "}
      <a href="/track-record" className="underline underline-offset-2 hover:text-pr-text">
        Track record
      </a>
    </p>
  );
}
