import { useEffect, useState } from "react";
import { api, type TrackRecord, type VsMarket } from "../api/client";
import { EmptyState, ErrorState, Skeleton, StatTable, kickoff } from "../predictor-ui";
import {
  Block,
  CalibrationChart,
  Disagreements,
  RecentPicks,
  RecordHeadline,
  WeekStrip,
  type Bucket,
  type TallyRow,
} from "./trackRecordViews";

// Plain names for each tracked market, in the order they read best.
const MARKETS: Record<string, string> = {
  game_outcome: "Winner pick",
  h2h: "Moneyline",
  spread: "Spread",
  total: "Total points",
  player_props: "Player props",
};
const ORDER = Object.keys(MARKETS);
const rank = (market: string) => (ORDER.includes(market) ? ORDER.indexOf(market) : ORDER.length);
const labelFor = (market: string) => MARKETS[market] ?? market;

const n0 = (x: number) => x.toLocaleString("en-US");

type RowWithBuckets = TrackRecord & { confidence_buckets?: Bucket[] };

/**
 * The track record, led by the only number that cannot flatter the model: picks
 * made before tip-off, counted once per game and market. Everything the model
 * "got right" after the game started is a rebuilt backtest, and lives in its own
 * block that never feeds the headline.
 */
export default function TrackRecordPanel() {
  const [rows, setRows] = useState<TrackRecord[] | null>(null);
  const [vsMarket, setVsMarket] = useState<VsMarket | null>(null);
  const [error, setError] = useState(false);
  const [vsError, setVsError] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    setError(false);
    setVsError(false);
    setRows(null);
    setVsMarket(null);
    // Two requests, each with its own failure: a broken comparison block must
    // not take the record above it down with it.
    api.getTrackRecord().then(setRows).catch(() => setError(true));
    api.getVsMarket().then(setVsMarket).catch(() => setVsError(true));
  }, [reloadKey]);

  if (error) return <ErrorState message="We couldn't load the track record. Check your connection and try again." onRetry={() => setReloadKey((k) => k + 1)} />;
  if (rows === null) return <Skeleton label="Loading track record…" />;
  if (rows.length === 0) return <EmptyState message="No tracked predictions yet. Picks are graded here once their games are final." />;

  const settled = [...rows].filter((r) => r.settled).sort((a, b) => rank(a.market) - rank(b.market));
  const unsettled = rows.filter((r) => !r.settled);
  const lead = settled.find((r) => r.market === "game_outcome") ?? settled[0];

  const tally: TallyRow[] = settled
    .filter((r) => r.pre_tip)
    .map((r) => ({ key: r.market, label: labelFor(r.market), correct: r.pre_tip!.correct_predictions, n: r.pre_tip!.total_predictions }))
    // The lead market first; the rest in reading order.
    .sort((a, b) => (a.key === lead?.market ? -1 : b.key === lead?.market ? 1 : 0));

  // Backtest: everything counted but made at or after tip-off. Hit rate only: these rows carry no stated confidence.
  const backtest = settled
    .filter((r) => (r.n_rebuilt ?? 0) > 0 && r.pre_tip)
    .map((r) => ({
      key: r.market,
      label: labelFor(r.market),
      n: r.n_rebuilt ?? 0,
      correct: r.correct_predictions - r.pre_tip!.correct_predictions,
    }));

  const weeks = lead?.pre_tip?.weekly ?? lead?.weekly ?? [];
  const calibration = (settled as RowWithBuckets[])
    .filter((r) => r.confidence_buckets?.length)
    .map((r) => ({ key: r.market, label: labelFor(r.market), buckets: r.confidence_buckets! }));
  const allPicks = rows.flatMap((r) => (r.per_pick ?? []).map((p) => ({ ...p, market: r.market })));
  const h2hPicks = rows.find((r) => r.market === "h2h")?.per_pick ?? [];
  const pushes = rows.reduce((n, r) => n + (r.n_push ?? 0), 0);

  return (
    <div className="flex flex-col gap-8">
      <Block id="tr-record" title="Counted record: picks made before tip-off">
        {tally.length === 0 ? (
          <p className="text-sm text-pr-text-dim">No pick made before tip-off has been graded yet.</p>
        ) : (
          <RecordHeadline rows={tally} />
        )}
      </Block>

      <div className="grid gap-8 lg:grid-cols-2">
        <Block id="tr-week" title="Week by week">
          <WeekStrip weeks={weeks} />
        </Block>
        <Block id="tr-calibration" title="Does 70% mean 70%?">
          <CalibrationChart series={calibration} />
        </Block>
      </div>

      <div className="grid gap-8 lg:grid-cols-2">
        <Block id="tr-market" title="Against the odds">
          {vsError ? (
            <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-pr border border-pr-loss/50 bg-pr-panel px-4 py-3 text-sm text-pr-text">
              <span>We couldn&apos;t load the comparison with the odds.</span>
              <button type="button" onClick={() => setReloadKey((k) => k + 1)} className="rounded-pr border border-pr-rule bg-pr-panel-2 px-3 py-1.5 text-xs font-semibold text-pr-text hover:border-pr-accent">
                Try again
              </button>
            </div>
          ) : vsMarket === null ? (
            <Skeleton label="Loading comparison…" />
          ) : vsMarket.n === 0 ? (
            <p className="text-sm text-pr-text-dim">No finished game has a pre-tip price to compare with yet.</p>
          ) : (
            <Disagreements
              gameIds={vsMarket.disagreement_game_ids}
              picks={h2hPicks}
              n={vsMarket.disagreement_n}
              hitRate={vsMarket.disagreement_hit_rate}
            />
          )}
        </Block>
        <Block id="tr-recent" title="Latest counted picks">
          <RecentPicks picks={allPicks} labelFor={labelFor} />
        </Block>
      </div>

      {backtest.length > 0 && (
        <Block id="tr-backtest" title="Backtest, not counted">
          <p className="max-w-prose text-sm text-pr-text-dim">
            Picks rebuilt at or after tip-off by a later run of the model. They show what the model would have said, not
            what it called in time, so they never enter the record above.
          </p>
          <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {backtest.map((b) => (
              <li key={b.key} data-testid={`backtest-${b.key}`} className="rounded-pr border border-pr-rule bg-pr-panel px-4 py-3">
                <p className="text-xs text-pr-text-dim">{b.label}</p>
                <p className="font-pr-display text-lg font-semibold tabular-nums text-pr-text">
                  {n0(b.correct)} <span className="text-pr-text-faint">of</span> {n0(b.n)}
                </p>
              </li>
            ))}
          </ul>
        </Block>
      )}

      <details className="rounded-pr border border-pr-rule bg-pr-panel px-4 py-3 text-sm">
        <summary className="cursor-pointer font-semibold text-pr-text">Every recorded pick</summary>
        <div className="mt-3">
          {allPicks.length === 0 ? (
            <p className="text-pr-text-dim">No pick-level data in this response.</p>
          ) : (
            <StatTable
              rows={[...allPicks].sort((a, b) => b.created_at.localeCompare(a.created_at))}
              rowKey={(r) => `${r.market}-${r.game_id}-${r.created_at}`}
              caption="Every recorded pick"
              columns={[
                { key: "gameday", label: "Game", value: (r) => r.gameday ?? r.created_at, render: (r) => (r.gameday ? kickoff(r.gameday) : <span className="text-pr-text-dim">Unknown</span>) },
                { key: "market", label: "Market", value: (r) => r.market, render: (r) => labelFor(r.market) },
                { key: "pick", label: "Pick", value: (r) => r.pick },
                { key: "actual", label: "Actual", value: (r) => r.actual },
                {
                  key: "hit",
                  label: "Result",
                  value: (r) => (r.hit === null ? 2 : r.hit ? 1 : 0),
                  render: (r) => (r.hit === null ? <span className="text-pr-text-faint">Push</span> : r.hit ? <span className="font-semibold text-pr-win">Right</span> : <span className="font-semibold text-pr-loss">Wrong</span>),
                },
                {
                  key: "timing",
                  label: "Made",
                  value: (r) => (r.made_before_tip ? 1 : r.counted ? 0 : -1),
                  render: (r) => (r.made_before_tip ? "Before tip-off" : r.counted ? "After tip-off" : "Rebuilt (not counted)"),
                },
              ]}
            />
          )}
        </div>
      </details>

      <details className="rounded-pr border border-pr-rule bg-pr-panel px-4 py-3 text-sm">
        <summary className="cursor-pointer font-semibold text-pr-text">How this is counted</summary>
        <div className="mt-3 flex max-w-prose flex-col gap-2 text-pr-text-dim">
          <p>
            One counted pick per game and market: the earliest recorded. A re-run of the model never replaces it and is
            never counted twice. The counted record above only includes picks made before tip-off.
          </p>
          {pushes > 0 && <p>{n0(pushes)} pick{pushes === 1 ? "" : "s"} landed on the line and {pushes === 1 ? "is" : "are"} left out of every rate, not scored as a miss.</p>}
          {unsettled.length > 0 && (
            <p>
              {unsettled.map((r) => labelFor(r.market)).join(", ")} {unsettled.length === 1 ? "has" : "have"} no results rule yet, so no rate is shown.
            </p>
          )}
          {vsMarket && Object.keys(vsMarket.method).length > 0 && (
            <dl className="mt-1 flex flex-col gap-2">
              {Object.entries(vsMarket.method).map(([k, v]) => (
                <div key={k}>
                  <dt className="text-xs font-semibold uppercase tracking-wide text-pr-text-faint">{k.replace(/_/g, " ")}</dt>
                  <dd className="text-xs leading-relaxed">{String(v)}</dd>
                </div>
              ))}
            </dl>
          )}
        </div>
      </details>
    </div>
  );
}
