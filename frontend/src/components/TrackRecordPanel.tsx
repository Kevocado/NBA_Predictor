import { useEffect, useState, type ReactNode } from "react";
import { api, type TrackRecord, type TrackRecordWeek, type VsMarket } from "../api/client";
import {
  EmptyState,
  ErrorState,
  Skeleton,
  StatTable,
  StatTile,
  kickoff,
  pct,
  record,
  signed,
  type Column,
} from "../predictor-ui";

// Plain names for each tracked market, in the order they read best.
const MARKETS: Record<string, string> = {
  game_outcome: "Winner pick",
  h2h: "Moneyline vs the market",
  spread: "Spread",
  total: "Total points",
  player_props: "Player props",
};
// Shorter labels for the weekly table's columns, same order.
const SHORT: Record<string, string> = {
  game_outcome: "Winner",
  h2h: "Moneyline",
  spread: "Spread",
  total: "Total",
  player_props: "Props",
};
const ORDER = Object.keys(MARKETS);
const rank = (market: string) => (ORDER.includes(market) ? ORDER.indexOf(market) : ORDER.length);
const labelFor = (market: string) => MARKETS[market] ?? market;
const shortFor = (market: string) => SHORT[market] ?? market;

const plural = (n: number, word: string) => `${n.toLocaleString("en-US")} ${word}${n === 1 ? "" : "s"}`;

/** One week of the weekly table: the week, its total, and one cell per market. */
type WeekRow = {
  week_start: string;
  picks: number;
  cells: Record<string, TrackRecordWeek | undefined>;
};

/**
 * The accuracy scale, and nothing else on it.
 *
 * The fill is the accuracy and ONLY the accuracy: no share of the week's
 * games is folded in, because a week with fewer games drawing a shorter bar
 * at a higher accuracy is the bug B2 exists to fix (a 1-game perfect week
 * would lose to a 4-game 50% week). Volume is the `n` in the text beside it.
 * The 50% marker sits at the midpoint of the scale so a reader can see which
 * side of even a week landed. aria-hidden: the figure beside it is real text,
 * and two copies of one number is worse for a screen reader than one.
 */
function AccuracyBar({ value, testId = "accuracy-bar" }: { value: number | null | undefined; testId?: string }) {
  const width = value == null || !Number.isFinite(value) ? 0 : Math.min(100, Math.max(0, value * 100));
  return (
    <span aria-hidden="true" data-testid={testId} className="relative block h-1.5 w-full overflow-hidden rounded-pr bg-pr-panel-2">
      <span className="absolute inset-y-0 left-0 bg-pr-accent" style={{ width: `${width}%` }} />
      <span data-testid="accuracy-50-marker" className="absolute inset-y-0 left-1/2 w-px bg-pr-text-faint" />
    </span>
  );
}

function AccuracyCard({ label, accuracy, graded, testId }: {
  label: string;
  accuracy: number | null;
  /** The record behind the rate, e.g. "58/100" — the n travels with it. */
  graded: string;
  testId: string;
}) {
  return (
    <div data-testid="accuracy-card" className="flex flex-col gap-1.5 rounded-pr border border-pr-rule bg-pr-panel p-4">
      <span className="text-xs font-semibold uppercase tracking-wide text-pr-text-dim">{label}</span>
      <span className="font-pr-display text-2xl font-semibold text-pr-text">{pct(accuracy ?? Number.NaN)}</span>
      <AccuracyBar value={accuracy} testId={testId} />
      <span className="text-xs tabular-nums text-pr-text-faint">{graded}</span>
    </div>
  );
}

function Section({ id, title, blurb, children }: {
  id: string;
  title: string;
  blurb?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section id={id} aria-labelledby={`${id}-heading`} className="flex flex-col gap-3">
      <div>
        <h3 id={`${id}-heading`} className="font-pr-display text-sm font-semibold uppercase tracking-wider text-pr-text-dim">
          {title}
        </h3>
        {blurb && <p className="mt-1 max-w-3xl text-xs leading-relaxed text-pr-text-dim">{blurb}</p>}
      </div>
      {children}
    </section>
  );
}

/** One market's cell in the weekly table: a rate with its count, or the gap. */
function WeekCell({ week }: { week: TrackRecordWeek | undefined }) {
  if (!week || week.n === 0) return <span data-testid="not-tracked">Not tracked</span>;
  return (
    <span className="tabular-nums">
      {pct(week.hit_rate ?? Number.NaN)} <span className="text-pr-text-faint">({week.n})</span>
    </span>
  );
}

/**
 * The week rows, one per week across every settled market.
 *
 * Weeks are the UNION of what each market sent, so a market that started
 * grading later does not shift the rows of the one beside it; a market with
 * no row for a week reads as "Not tracked" in its own cell. The backend
 * sends the same window to every settled market, but the join here is on
 * week_start rather than on an index, so a payload that ever stops doing
 * that degrades to a visible gap instead of a silently wrong column.
 */
function buildWeeklyRows(rows: TrackRecord[]): WeekRow[] {
  const weekStarts = [...new Set(rows.flatMap((row) => (row.weekly ?? []).map((w) => w.week_start)))].sort();
  if (weekStarts.length === 0) return [];
  return weekStarts.map((week_start) => {
    const cells: Record<string, TrackRecordWeek | undefined> = {};
    let picks = 0;
    for (const row of rows) {
      const cell = (row.weekly ?? []).find((w) => w.week_start === week_start);
      cells[row.market] = cell;
      if (cell) picks += cell.n;
    }
    return { week_start, picks, cells };
  });
}

/** The heading for each method key the backend sends — a LABEL MAP, not a
 *  list of keys to render: an unknown key still gets a heading built from its
 *  own name, so a new sentence reaches the page instead of being dropped. */
const METHOD_LABELS: Record<string, string | undefined> = {
  market_probability: "Where the market number comes from",
  edge: "What edge means",
  disagreement: "The disagreement cohort",
  not_a_profit_claim: "What this is not",
  population: "Who is in this number",
};
const methodLabel = (key: string) => METHOD_LABELS[key] ?? key.replace(/_/g, " ");

/** A section whose data this response does not carry, said plainly. */
function NotRecorded({ why }: { why: string }) {
  return (
    <p data-testid="not-recorded" className="rounded-pr border border-pr-rule bg-pr-panel px-3 py-2 text-sm text-pr-text-dim">
      {why}
    </p>
  );
}

function VsMarketSection({ vsMarket, error, onRetry }: {
  vsMarket: VsMarket | null;
  error: boolean;
  onRetry: () => void;
}) {
  if (error) {
    return (
      <Section id="tr-market" title="Vs the market" blurb="The model's probability beside the price it was measured against.">
        <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-pr border border-pr-loss/50 bg-pr-panel px-4 py-3 text-sm text-pr-text">
          <span>We couldn&apos;t load the model-versus-market comparison.</span>
          <button type="button" onClick={onRetry} className="rounded-pr border border-pr-rule bg-pr-panel-2 px-3 py-1.5 text-xs font-semibold text-pr-text hover:border-pr-accent">
            Try again
          </button>
        </div>
      </Section>
    );
  }
  if (!vsMarket) return null;

  const { scope, method } = vsMarket;
  const methodRows = Object.entries(method);

  return (
    <Section
      id="tr-market"
      title="Vs the market"
      blurb={
        <>
          The model&apos;s moneyline probability beside the price the book carried for the same side, over the
          games where both exist. This is agreement with a price, not a return: nothing in this section is a
          stake, a yield or a cent.
        </>
      }
    >
      {vsMarket.n === 0 ? (
        <NotRecorded why="No finished game has a pre-tip moneyline price to compare with yet." />
      ) : (
        <>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <StatTile label="Games compared" value={vsMarket.n.toLocaleString("en-US")} sub="finished, priced before tip-off" />
            <StatTile
              label="Model says"
              value={pct(vsMarket.mean_model_probability ?? Number.NaN)}
              sub={`for its pick · over ${plural(vsMarket.n, "game")}`}
            />
            <StatTile
              label="Price says"
              value={pct(vsMarket.mean_market_probability ?? Number.NaN)}
              sub={`for that side · over ${plural(vsMarket.n, "game")}`}
            />
            <StatTile
              label="Mean edge"
              value={vsMarket.mean_edge_points == null ? "—" : `${signed(vsMarket.mean_edge_points)} pt`}
              sub={<>model minus price · over {plural(vsMarket.n, "game")}</>}
            />
          </div>

          {/* The cohort is the number to read: a pick against the price, and
              how often it landed. The mean edge above is a calibration check,
              and the backend's own words say so. */}
          <div className="rounded-pr border border-pr-rule bg-pr-panel p-4">
            <h4 className="font-pr-display text-xs font-semibold uppercase tracking-wider text-pr-text-faint">
              Games where the model backed the side the price did not
            </h4>
            <p className="mt-1 font-pr-display text-2xl font-semibold text-pr-text">
              {pct(vsMarket.disagreement_hit_rate ?? Number.NaN)}
            </p>
            <p className="text-xs tabular-nums text-pr-text-dim">
              Hit rate over {plural(vsMarket.disagreement_n, "game")} where it disagreed with the price.{" "}
              {vsMarket.disagreement_n === 0 && "A 50/50 price favours nobody, so those games are compared but left out of this cohort."}
            </p>
            {vsMarket.disagreement_game_ids.length > 0 && (
              <p className="mt-2 break-words text-xs text-pr-text-faint">
                Game IDs in this cohort:{" "}
                <span className="tabular-nums text-pr-text-dim">{vsMarket.disagreement_game_ids.join(", ")}</span>
              </p>
            )}
          </div>

          {scope.n_games_in_weekly + scope.n_games_outside_weekly === scope.n_games_total && (
            <p data-testid="scope-note" className="max-w-3xl text-xs leading-relaxed text-pr-text-dim">
              The figures above cover <span className="tabular-nums">{scope.n_games_total.toLocaleString("en-US")}</span>{" "}
              games — {scope.population}
              {scope.weekly_from && scope.weekly_through && (
                <>
                  . The week table below accounts for{" "}
                  <span className="tabular-nums">{scope.n_games_in_weekly.toLocaleString("en-US")}</span> of them, from{" "}
                  <span className="tabular-nums">{kickoff(scope.weekly_from)}</span> through{" "}
                  <span className="tabular-nums">{kickoff(scope.weekly_through)}</span>
                </>
              )}
              {scope.n_games_outside_weekly === 0
                ? ". Every game in the headline is in the table."
                : `. ${plural(scope.n_games_outside_weekly, "game")} in the headline ${
                    scope.n_games_outside_weekly === 1 ? "falls" : "fall"
                  } outside that window, which is why the table does not add up to the count above.`}
            </p>
          )}

          {vsMarket.weekly.length > 0 && (
            <StatTable
              rows={vsMarket.weekly}
              rowKey={(r) => r.week_start}
              caption="Model against the price, by week"
              columns={[
                { key: "week", label: "Week", value: (r) => r.week_start, render: (r) => kickoff(r.week_start) },
                {
                  key: "n",
                  label: "Compared",
                  numeric: true,
                  value: (r) => r.n,
                  render: (r) => (r.n > 0 ? <span className="tabular-nums">{r.n}</span> : <span data-testid="not-compared">Not compared</span>),
                },
                {
                  key: "edge",
                  label: "Mean edge",
                  numeric: true,
                  firstDir: "asc",
                  value: (r) => r.mean_edge_points,
                  render: (r) => (
                    <span className="tabular-nums">{r.mean_edge_points == null ? "—" : `${signed(r.mean_edge_points)} pt`}</span>
                  ),
                },
                {
                  key: "cohort",
                  label: "Cohort",
                  firstDir: "desc",
                  value: (r) => r.disagreement_hit_rate,
                  render: (r) =>
                    r.disagreement_n > 0 ? (
                      <span className="tabular-nums">
                        {pct(r.disagreement_hit_rate ?? Number.NaN)} <span className="text-pr-text-faint">({r.disagreement_n})</span>
                      </span>
                    ) : (
                      <span data-testid="not-compared">No disagreement</span>
                    ),
                },
              ]}
            />
          )}

          {/* Printed from the payload, verbatim, once per key it sent: the
              backend owns this wording and the page does not paraphrase it. */}
          {methodRows.length > 0 && (
            <dl data-testid="method-block" className="flex flex-col gap-2 rounded-pr border border-pr-rule bg-pr-panel p-4">
              {methodRows.map(([key, value]) => (
                <div key={key} data-testid="method-row">
                  <dt className="text-xs font-semibold uppercase tracking-wide text-pr-text-faint">{methodLabel(key)}</dt>
                  <dd className="mt-0.5 max-w-3xl text-xs leading-relaxed text-pr-text-dim">{String(value)}</dd>
                </div>
              ))}
            </dl>
          )}
        </>
      )}
    </Section>
  );
}

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

  const sorted = [...rows].sort((a, b) => rank(a.market) - rank(b.market));
  const settled = sorted.filter((row) => row.settled);
  const unsettled = sorted.filter((row) => !row.settled);
  // The headline counts every recorded pick, whenever it was made, so
  // `n_rebuilt` is no longer what was withheld -- it is how much of the
  // headline is the rerun rather than the night, and the reconciliation between
  // the two figures. `n_pre_tip` is the honest read of live performance.
  const afterTip = rows.reduce((n, r) => n + (r.n_rebuilt ?? 0), 0);
  const preTip = rows.reduce((n, r) => n + (r.n_pre_tip ?? r.pre_tip?.total_predictions ?? 0), 0);
  const graded = rows.reduce((n, r) => n + r.total_predictions, 0);
  const pushes = rows.reduce((n, r) => n + (r.n_push ?? 0), 0);
  const unplaced = rows.reduce((n, r) => n + (r.n_unplaced ?? 0), 0);
  // Only rows that actually carry a pre-tip figure: an unsettled market has no
  // rule to apply one with, and a row with none is not a zero.
  const preTipRows = settled.filter((row) => row.pre_tip);
  const weeklyRows = buildWeeklyRows(settled);
  // The bar column: the model's own winner call, else the first settled
  // market. One bar per row -- four would be a wall, not a comparison.
  const barMarket = settled.find((r) => r.market === "game_outcome")?.market ?? settled[0]?.market;

  const weeklyColumns: Column<WeekRow>[] = [
    { key: "week", label: "Week", value: (r) => r.week_start, render: (r) => kickoff(r.week_start) },
    { key: "picks", label: "Picks", numeric: true, value: (r) => r.picks, render: (r) => <span className="tabular-nums">{r.picks}</span> },
    ...settled.map<Column<WeekRow>>((row) => ({
      key: row.market,
      label: shortFor(row.market),
      firstDir: "desc",
      value: (r) => r.cells[row.market]?.hit_rate ?? null,
      render: (r) => {
        const cell = r.cells[row.market];
        if (row.market !== barMarket) return <WeekCell week={cell} />;
        if (!cell || cell.n === 0) return <WeekCell week={cell} />;
        return (
          <span className="flex items-center gap-2">
            <span className="inline-block w-16 shrink-0">
              <AccuracyBar value={cell.hit_rate} testId="week-bar" />
            </span>
            <WeekCell week={cell} />
          </span>
        );
      },
    })),
  ];

  return (
    <div className="flex flex-col gap-6">
      <p className="max-w-prose text-sm text-pr-text-dim">
        Every recorded pick counts, whenever it was made. When a pick was made
        still matters, so the figure beside each rate is the pre-tip subset with
        its own count.
        {afterTip > 0 &&
          ` ${graded.toLocaleString("en-US")} picks are graded here: ${preTip.toLocaleString("en-US")} were made before their game tipped off, and ${afterTip.toLocaleString("en-US")} at or after it, on a re-run of the model.`}
      </p>

      <Section
        id="tr-record"
        title="Record"
        blurb="How good the model has been, in aggregate. Every rate carries the number of picks behind it, and one counted pick per game and market — the first one recorded, so a re-run of the model neither replaces it nor is counted twice."
      >
        <p data-testid="timing-note" className="max-w-3xl rounded-pr border border-pr-rule bg-pr-panel px-3 py-2 text-sm text-pr-text-dim">
          {afterTip === 0
            ? "Every pick in this record was made before its game tipped off, so the pre-tip table below is the same number over the same picks."
            : `${afterTip.toLocaleString("en-US")} of these ${graded.toLocaleString("en-US")} picks were recorded at or after their own tip-off — a re-run of the model, not a pre-game call. They are counted, and every pick row the API sends carries its own timestamp and says which it was.`}
          {pushes > 0 && ` ${plural(pushes, "pick")} landed on the line and ${pushes === 1 ? "was" : "were"} left out of the rate, not scored as a miss.`}
          {unplaced > 0 && ` ${plural(unplaced, "pick")} ${unplaced === 1 ? "has" : "have"} no game date, so ${unplaced === 1 ? "it appears" : "they appear"} in no week below.`}
        </p>

        {settled.length === 0 ? (
          <NotRecorded why="None of these markets can be judged against results yet." />
        ) : (
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            {settled.map((row) => (
              <AccuracyCard
                key={row.market}
                label={`${labelFor(row.market)} accuracy`}
                accuracy={row.total_predictions > 0 ? row.hit_rate : null}
                graded={record(row.correct_predictions, row.total_predictions)}
                testId={`accuracy-bar-${row.market}`}
              />
            ))}
          </div>
        )}

        {preTipRows.length > 0 && (
          <div className="mt-4">
            <h3 className="mb-2 font-pr-display text-sm font-semibold uppercase tracking-wide text-pr-text-dim">
              Made before tip-off
            </h3>
            <p className="mb-2 max-w-prose text-xs leading-relaxed text-pr-text-dim">
              The same markets over the picks that were made before their own
              tip-off — what the model would have said on the night. Read this
              for live performance: the cards above also count picks taken on a
              re-run afterwards, and the two figures differ by exactly the number
              named above.
            </p>
            <StatTable
              rows={preTipRows}
              rowKey={(r) => r.market}
              caption="Accuracy on picks made before tip-off"
              columns={[
                { key: "market", label: "Market", value: (r) => r.market, render: (r) => labelFor(r.market) },
                {
                  key: "picks", label: "Picks", numeric: true,
                  value: (r) => r.pre_tip?.total_predictions ?? 0,
                  render: (r) => <span className="tabular-nums">{(r.pre_tip?.total_predictions ?? 0).toLocaleString("en-US")}</span>,
                },
                {
                  key: "hits", label: "Record", numeric: true,
                  value: (r) => r.pre_tip?.total_predictions ?? 0,
                  render: (r) => (r.pre_tip && r.pre_tip.total_predictions > 0
                    ? <span className="tabular-nums">{record(r.pre_tip.correct_predictions, r.pre_tip.total_predictions)}</span>
                    : <span className="text-pr-text-faint">—</span>),
                },
                {
                  key: "rate", label: "Accuracy", numeric: true, firstDir: "desc",
                  value: (r) => (r.pre_tip?.total_predictions ? r.pre_tip.hit_rate : null),
                  render: (r) => (r.pre_tip && r.pre_tip.total_predictions > 0 && r.pre_tip.hit_rate !== null
                    ? pct(r.pre_tip.hit_rate)
                    : <span className="text-pr-text-faint">—</span>),
                },
              ]}
            />
          </div>
        )}

        {unsettled.length > 0 && (
          <p data-testid="unsettled-note" className="max-w-3xl rounded-pr border border-pr-rule bg-pr-panel px-3 py-2 text-sm text-pr-text-dim">
            {unsettled
              .map((row) => `${labelFor(row.market)}: ${row.total_predictions.toLocaleString("en-US")} stored rows, not settled yet`)
              .join(" · ")}
            . No rate is shown for them, because none has been measured.
          </p>
        )}
      </Section>

      <Section
        id="tr-week"
        title="By week"
        blurb={
          <>
            Every week since the tracker began, through this week — including the ones with nothing in them. Each
            market carries its own count, because a week can grade four winner picks and one spread.
          </>
        }
      >
        {weeklyRows.length === 0 ? (
          <NotRecorded why="No week-by-week record in this response." />
        ) : (
          <>
            <StatTable rows={weeklyRows} columns={weeklyColumns} rowKey={(r) => r.week_start} caption="Accuracy by week, per market" />
            <p className="max-w-3xl text-xs leading-relaxed text-pr-text-dim">
              The bar is the accuracy on its own scale, with the 50% line marked. The pick count is a separate
              number: a week with fewer picks is not a worse week, it is a thinner one.
            </p>
          </>
        )}
      </Section>

      <VsMarketSection vsMarket={vsMarket} error={vsError} onRetry={() => setReloadKey((k) => k + 1)} />

      {/* Per-pick detail table: every recorded pick, one row. */}
      <Section
        id="tr-picks"
        title="Every pick"
        blurb="One row per recorded pick, in the order they were made. The timing badge says whether it was made before tip-off. Only counted picks (the first recorded) contribute to the headline rate; reruns are shown for history but not scored."
      >
        {(() => {
          const allPicks = rows.flatMap((r) => (r.per_pick ?? []).map((p) => ({ ...p, market: r.market })));
          if (allPicks.length === 0) return <NotRecorded why="No pick-level data in this response." />;
          const sorted = [...allPicks].sort((a, b) => a.created_at.localeCompare(b.created_at));
          return (
            <StatTable
              rows={sorted}
              rowKey={(r) => `${r.market}-${r.game_id}-${r.created_at}`}
              caption="Every recorded pick for settled markets"
              columns={[
                { key: "gameday", label: "Game", value: (r) => r.gameday ?? r.created_at, render: (r) => r.gameday ? kickoff(r.gameday) : <span className="text-pr-text-dim">Unknown</span> },
                { key: "market", label: "Market", value: (r) => r.market, render: (r) => labelFor(r.market) },
                { key: "pick", label: "Pick", value: (r) => r.pick },
                { key: "actual", label: "Actual", value: (r) => r.actual },
                {
                  key: "hit",
                  label: "Result",
                  value: (r) => (r.hit === null ? 2 : r.hit ? 1 : 0),
                  render: (r) => r.hit === null ? (
                    <span className="text-pr-text-faint">No result</span>
                  ) : r.hit ? (
                    <span className="text-pr-win font-semibold">✓</span>
                  ) : (
                    <span className="text-pr-loss font-semibold">✗</span>
                  ),
                },
                {
                  key: "timing",
                  label: "Made",
                  value: (r) => (r.made_before_tip ? 1 : r.counted ? 0 : -1),
                  render: (r) => r.made_before_tip ? (
                    <span className="text-pr-win font-semibold">Before tip-off</span>
                  ) : r.counted ? (
                    <span className="text-pr-text-faint">After tip-off</span>
                  ) : (
                    <span className="text-pr-text-dim">Rebuilt (not counted)</span>
                  ),
                },
                { key: "when", label: "When", value: (r) => r.created_at, render: (r) => kickoff(r.created_at) },
              ]}
            />
          );
        })()}
      </Section>

      {/* Biggest upsets and biggest misses tables. */}
      <Section
        id="tr-extremes"
        title="Biggest upsets & biggest misses"
        blurb="The counted picks where the model was confident and wrong (misses), and where the market was confident and wrong (upsets — the model backed the winner and the market didn't). Placeholder sort by gameday; full sort by model probability needs pick probability in per_pick."
      >
        {(() => {
          const allPicks = rows.flatMap((r) => (r.per_pick ?? []).map((p) => ({ ...p, market: r.market })));
          const counted = allPicks.filter((p) => p.counted && p.hit !== null);
          if (counted.length === 0) return <NotRecorded why="No counted picks graded yet." />;

          const misses = counted.filter((p) => p.hit === false);
          const upsets = counted.filter((p) => p.hit === true);

          return (
            <div className="grid gap-4 sm:grid-cols-2">
              <StatTable
                rows={misses.slice().sort((a, b) => a.created_at.localeCompare(b.created_at)).slice(0, 10)}
                rowKey={(r) => `${r.market}-${r.game_id}-${r.created_at}`}
                caption="Biggest misses (model wrong)"
                columns={[
                  { key: "gameday", label: "Game", value: (r) => r.gameday ?? r.created_at, render: (r) => r.gameday ? kickoff(r.gameday) : <span className="text-pr-text-dim">Unknown</span> },
                  { key: "market", label: "Market", value: (r) => r.market, render: (r) => labelFor(r.market) },
                  { key: "pick", label: "Pick", value: (r) => r.pick },
                  { key: "actual", label: "Actual", value: (r) => r.actual },
                  { key: "when", label: "When", value: (r) => r.created_at, render: (r) => kickoff(r.created_at) },
                ]}
              />
              <StatTable
                rows={upsets.slice().sort((a, b) => a.created_at.localeCompare(b.created_at)).slice(0, 10)}
                rowKey={(r) => `${r.market}-${r.game_id}-${r.created_at}`}
                caption="Biggest upsets (market wrong, model right)"
                columns={[
                  { key: "gameday", label: "Game", value: (r) => r.gameday ?? r.created_at, render: (r) => r.gameday ? kickoff(r.gameday) : <span className="text-pr-text-dim">Unknown</span> },
                  { key: "market", label: "Market", value: (r) => r.market, render: (r) => labelFor(r.market) },
                  { key: "pick", label: "Pick", value: (r) => r.pick },
                  { key: "actual", label: "Actual", value: (r) => r.actual },
                  { key: "when", label: "When", value: (r) => r.created_at, render: (r) => kickoff(r.created_at) },
                ]}
              />
            </div>
          );
        })()}
      </Section>

      {/* Projected final standings panel. */}
      <Section
        id="tr-projected"
        title="Projected final standings"
        blurb="Each team's current win percentage and games back, projected to 82 games. The model's win probability on every remaining game is summed to produce a projected win total."
      >
        <NotRecorded why="Projected standings endpoint not yet implemented." />
      </Section>
    </div>
  );
}
