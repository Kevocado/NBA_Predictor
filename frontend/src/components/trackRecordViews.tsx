import { useState, type ReactNode } from "react";
import type { TrackRecordPick, TrackRecordWeek } from "../api/client";
import { kickoff, pct } from "../predictor-ui";

/**
 * Wilson 95% interval for a hit rate. With a few dozen counted picks the honest
 * statement is a range, not a point: 11 of 24 is "somewhere between 27% and 65%".
 */
export function wilson(correct: number, n: number): [number, number] | null {
  if (n <= 0) return null;
  const z = 1.96;
  const p = correct / n;
  const denom = 1 + (z * z) / n;
  const centre = (p + (z * z) / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

const whole = (x: number) => `${Math.round(x * 100)}%`;

/** Below this many counted picks the page says so, instead of letting a rate look settled. */
export const EARLY_SAMPLE = 30;

export interface TallyRow {
  key: string;
  label: string;
  correct: number;
  n: number;
}

/** One counted-record line: "5 of 10", and a bar showing the range that sample allows. */
function TallyLine({ row, lead }: { row: TallyRow; lead?: boolean }) {
  const range = wilson(row.correct, row.n);
  const rate = row.n > 0 ? row.correct / row.n : null;
  return (
    <div data-testid={`tally-${row.key}`} className="grid grid-cols-[minmax(7rem,9rem)_1fr] items-center gap-x-4 gap-y-1 sm:grid-cols-[minmax(7rem,9rem)_8rem_1fr]">
      <span className={lead ? "font-pr-display text-sm font-semibold text-pr-text" : "text-sm text-pr-text-dim"}>{row.label}</span>
      <span className={`tabular-nums ${lead ? "font-pr-display text-3xl font-semibold" : "text-lg font-semibold"} text-pr-text`}>
        {row.correct}
        <span className="text-pr-text-faint"> of </span>
        {row.n}
        {rate != null && <span className="ml-2 text-sm font-normal text-pr-text-dim">{whole(rate)}</span>}
      </span>
      <div className="col-span-2 sm:col-span-1">
        {range ? (
          <>
            <div
              role="img"
              aria-label={`Plausible range ${whole(range[0])} to ${whole(range[1])}, 50 percent marked`}
              className="relative h-2 rounded-pr bg-pr-panel-2"
            >
              <span
                className="absolute inset-y-0 rounded-pr bg-pr-accent/70"
                style={{ left: `${range[0] * 100}%`, width: `${Math.max(1, (range[1] - range[0]) * 100)}%` }}
              />
              {rate != null && (
                <span className="absolute -inset-y-0.5 w-0.5 bg-pr-text" style={{ left: `calc(${rate * 100}% - 1px)` }} />
              )}
              <span className="absolute -inset-y-1 left-1/2 w-px bg-pr-text-faint" />
            </div>
            <p className="mt-1 text-xs tabular-nums text-pr-text-faint">
              Could plausibly be {whole(range[0])}–{whole(range[1])}
            </p>
          </>
        ) : (
          <span className="text-xs text-pr-text-faint">Nothing graded yet</span>
        )}
      </div>
    </div>
  );
}

/** The counted record: picks made before tip-off. Each line carries its own sample size. */
export function RecordHeadline({ rows }: { rows: TallyRow[] }) {
  if (rows.length === 0) return null;
  const [lead, ...rest] = rows;
  return (
    <div className="flex flex-col gap-4 rounded-pr border border-pr-rule bg-pr-panel p-5">
      <TallyLine row={lead} lead />
      {lead.n < EARLY_SAMPLE && (
        <p data-testid="early-sample" className="text-sm text-pr-text-dim">
          Early sample: {lead.n} counted picks is not enough to call this model good or bad. The bar shows the range the
          true rate could sit in; the thin line is 50%.
        </p>
      )}
      {rest.length > 0 && (
        <div className="flex flex-col gap-3 border-t border-pr-rule pt-4">
          {rest.map((r) => (
            <TallyLine key={r.key} row={r} />
          ))}
        </div>
      )}
    </div>
  );
}

function weekLabel(iso: string) {
  return kickoff(iso);
}

/** One cell per week: colour is the hit rate, height is how many picks stood behind it. */
export function WeekStrip({ weeks }: { weeks: TrackRecordWeek[] }) {
  // Only weeks from the first tracked one on: the empty history before tracking began is not a record.
  const first = weeks.findIndex((w) => w.tracked && w.n > 0);
  const shown = first === -1 ? [] : weeks.slice(first);
  const maxN = Math.max(1, ...shown.map((w) => w.n));
  if (shown.length === 0) {
    return <p className="text-sm text-pr-text-dim">No graded week yet.</p>;
  }
  return (
    <div>
      <ol className="flex h-24 items-end gap-1" aria-label="Hit rate by week">
        {shown.map((w) => {
          const graded = w.tracked && w.n > 0 && w.hit_rate != null;
          const h = graded ? 18 + (w.n / maxN) * 82 : 12;
          const tone = !graded ? "bg-pr-panel-2" : w.hit_rate! >= 0.5 ? "bg-pr-win" : "bg-pr-loss";
          return (
            <li
              key={w.week_start}
              data-testid="week-cell"
              title={graded ? `${weekLabel(w.week_start)}: ${w.correct} of ${w.n} (${whole(w.hit_rate!)})` : `${weekLabel(w.week_start)}: nothing graded`}
              className={`min-w-1.5 max-w-9 flex-1 rounded-t-pr ${tone} ${graded ? "" : "opacity-60"}`}
              style={{ height: `${h}%` }}
            >
              <span className="sr-only">
                {graded ? `${weekLabel(w.week_start)}: ${w.correct} of ${w.n}` : `${weekLabel(w.week_start)}: nothing graded`}
              </span>
            </li>
          );
        })}
      </ol>
      <p className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-pr-text-faint">
        <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-pr bg-pr-win" />50% or better</span>
        <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-pr bg-pr-loss" />under 50%</span>
        <span>Taller = more picks that week</span>
      </p>
    </div>
  );
}

export interface Bucket {
  bucket: string;
  total_predictions: number;
  correct_predictions: number;
  hit_rate: number | null;
}

// "50-60%" -> 0.55, "70%+" -> 0.8: where the bucket sits on the stated-confidence axis.
function bucketMid(label: string): number | null {
  const m = /(\d+)\s*-\s*(\d+)/.exec(label);
  if (m) return (Number(m[1]) + Number(m[2])) / 200;
  const plus = /(\d+)\s*%?\s*\+/.exec(label);
  return plus ? (Number(plus[1]) + 10) / 100 : null;
}

const THIN = 5;

/** Stated confidence against what actually happened. On the diagonal = trustworthy. */
export function CalibrationChart({ series }: { series: { key: string; label: string; buckets: Bucket[] }[] }) {
  const usable = series.filter((s) => s.buckets.some((b) => b.total_predictions > 0));
  const [pick, setPick] = useState(usable[0]?.key);
  if (usable.length === 0) return <p className="text-sm text-pr-text-dim">No confidence buckets graded yet.</p>;
  const active = usable.find((s) => s.key === pick) ?? usable[0];
  const W = 320, H = 220, L = 34, B = 26, T = 8, R = 8;
  const x = (c: number) => L + ((c - 0.45) / 0.45) * (W - L - R);
  const y = (v: number) => H - B - v * (H - B - T);
  const pts = active.buckets
    .map((b) => ({ b, mid: bucketMid(b.bucket) }))
    .filter((p): p is { b: Bucket; mid: number } => p.mid != null && p.b.total_predictions > 0 && p.b.hit_rate != null);
  return (
    <div>
      {usable.length > 1 && (
        <div role="group" aria-label="Market" className="mb-3 inline-flex rounded-pr border border-pr-rule p-0.5">
          {usable.map((s) => (
            <button
              key={s.key}
              type="button"
              aria-pressed={s.key === active.key}
              onClick={() => setPick(s.key)}
              className={`rounded-pr px-3 py-1 text-xs font-semibold ${s.key === active.key ? "bg-pr-panel-2 text-pr-text" : "text-pr-text-dim hover:text-pr-text"}`}
            >
              {s.label}
            </button>
          ))}
        </div>
      )}
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-md" role="img" aria-label={`Calibration for ${active.label}`}>
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} className="stroke-pr-rule" strokeWidth={1} />
            <text x={L - 6} y={y(v) + 4} textAnchor="end" className="fill-pr-text-faint text-[10px]">{Math.round(v * 100)}%</text>
          </g>
        ))}
        {/* The diagonal: stated confidence = actual hit rate. */}
        <line x1={x(0.45)} y1={y(0.45)} x2={x(0.9)} y2={y(0.9)} className="stroke-pr-text-faint" strokeDasharray="4 4" strokeWidth={1} />
        {[0.5, 0.6, 0.7, 0.8].map((c) => (
          <text key={c} x={x(c)} y={H - 8} textAnchor="middle" className="fill-pr-text-faint text-[10px]">{Math.round(c * 100)}%</text>
        ))}
        {pts.map(({ b, mid }) => {
          const thin = b.total_predictions < THIN;
          return (
            <circle
              key={b.bucket}
              data-testid="calibration-point"
              cx={x(mid)}
              cy={y(b.hit_rate!)}
              r={5 + Math.min(10, b.total_predictions)}
              className={thin ? "fill-pr-accent/25 stroke-pr-accent" : "fill-pr-accent/60 stroke-pr-accent"}
              strokeWidth={1.5}
            >
              <title>{`${b.bucket} confidence: ${b.correct_predictions} of ${b.total_predictions} right`}</title>
            </circle>
          );
        })}
      </svg>
      <p className="mt-1 text-xs text-pr-text-faint">
        Across: how sure the model said it was. Up: how often it was right. On the dashed line is honest; bigger dots
        rest on more picks; faint dots rest on fewer than {THIN}.
      </p>
    </div>
  );
}

const MARK = {
  win: <span className="font-semibold text-pr-win" aria-label="Right">Right</span>,
  loss: <span className="font-semibold text-pr-loss" aria-label="Wrong">Wrong</span>,
  push: <span className="text-pr-text-faint">Push</span>,
};
const markFor = (hit: boolean | null) => (hit === null ? MARK.push : hit ? MARK.win : MARK.loss);

/** The counted picks, newest first: what was called, what happened. */
export function RecentPicks({ picks, labelFor, limit = 12 }: { picks: (TrackRecordPick & { market: string })[]; labelFor: (m: string) => string; limit?: number }) {
  const rows = picks
    .filter((p) => p.counted && p.made_before_tip)
    .sort((a, b) => (b.gameday ?? b.created_at).localeCompare(a.gameday ?? a.created_at))
    .slice(0, limit);
  if (rows.length === 0) return <p className="text-sm text-pr-text-dim">No counted pick has been graded yet.</p>;
  return (
    <ul className="divide-y divide-pr-rule rounded-pr border border-pr-rule bg-pr-panel text-sm">
      {rows.map((p) => (
        <li key={`${p.market}-${p.game_id}-${p.created_at}`} data-testid="recent-pick" className="grid grid-cols-[4.5rem_1fr_auto] items-center gap-x-3 px-4 py-2">
          <span className="tabular-nums text-pr-text-faint">{p.gameday ? kickoff(p.gameday) : "—"}</span>
          <span className="min-w-0 truncate">
            <span className="font-semibold text-pr-text">{p.pick}</span>
            <span className="text-pr-text-faint"> · {labelFor(p.market)} · was {p.actual}</span>
          </span>
          {markFor(p.hit)}
        </li>
      ))}
    </ul>
  );
}

/** Games where the model backed the side the price did not, with how each one ended. */
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

/** A titled block. No blurb: the data under the title is the explanation. */
export function Block({ title, children, id }: { title: string; children: ReactNode; id: string }) {
  return (
    <section id={id} aria-labelledby={`${id}-h`} className="flex flex-col gap-3">
      <h3 id={`${id}-h`} className="font-pr-display text-base font-semibold text-pr-text">{title}</h3>
      {children}
    </section>
  );
}
