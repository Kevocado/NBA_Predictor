import { useEffect, useState } from "react";
import { api, type TrackRecord } from "../api/client";
import { EmptyState, ErrorState, Skeleton, pct } from "../predictor-ui";

// Plain names for each tracked market, in the order they read best.
const MARKETS: Record<string, string> = {
  game_outcome: "Winner pick",
  h2h: "Moneyline vs the market",
  spread: "Spread",
  total: "Total points",
};
// Only these are settled against results; the others' stored rows can't be
// judged yet, so they never show a made-up 0%.
const SETTLED = new Set(["game_outcome", "h2h"]);
const ORDER = Object.keys(MARKETS);
const rank = (market: string) => (ORDER.includes(market) ? ORDER.indexOf(market) : ORDER.length);

export default function TrackRecordPanel() {
  const [rows, setRows] = useState<TrackRecord[] | null>(null);
  const [error, setError] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    setError(false);
    setRows(null);
    api.getTrackRecord().then(setRows).catch(() => setError(true));
  }, [reloadKey]);

  if (error) return <ErrorState message="We couldn't load the track record. Check your connection and try again." onRetry={() => setReloadKey((k) => k + 1)} />;
  if (rows === null) return <Skeleton label="Loading track record…" />;
  if (rows.length === 0) return <EmptyState message="No tracked predictions yet. Picks are graded here once their games are final." />;

  const sorted = [...rows].sort((a, b) => rank(a.market) - rank(b.market));
  const rebuilt = rows.reduce((n, r) => n + (r.n_rebuilt ?? 0), 0);

  return (
    <div>
      <p className="mb-4 max-w-prose text-sm text-pr-text-dim">
        Only picks made before tip-off count.
        {rebuilt > 0 && ` ${rebuilt.toLocaleString("en-US")} finals had only a pick rebuilt after tip-off, so they are left out.`}
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-pr-text-dim">
              <th>Market</th>
              <th>Picks</th>
              <th>Correct</th>
              <th>Hit rate</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((row) => {
              const settled = SETTLED.has(row.market);
              return (
                <tr key={row.market}>
                  <td>{MARKETS[row.market] ?? row.market}</td>
                  <td>{row.total_predictions.toLocaleString("en-US")}</td>
                  <td>{settled ? row.correct_predictions.toLocaleString("en-US") : "Not settled yet"}</td>
                  <td>{settled && row.total_predictions > 0 ? pct(row.hit_rate) : settled ? "—" : "Not settled yet"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
