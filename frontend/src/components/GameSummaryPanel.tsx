import { useCallback, useEffect, useState } from "react";
import { ExplainerPanel, type Explanation } from "../predictor-ui";

/**
 * The "In plain English" panel on its own, so the game detail gets the same
 * behaviour as every other surface without re-implementing the fetch: request
 * the summary for this game, never block what is already on screen, and offer
 * a retry when it fails.
 *
 * `fetcher` defaults to the site's own client. A caller with no explainer
 * deployed (or a game the service has nothing for) passes nothing, and the
 * panel renders nothing rather than an empty box.
 */
export function GameSummaryPanel({
  gameId,
  fetcher,
  sport = "nba",
  className = "",
}: {
  gameId: string;
  fetcher?: (sport: string, id: string) => Promise<Explanation>;
  sport?: string;
  className?: string;
}) {
  const [data, setData] = useState<Explanation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);

  const load = useCallback(() => {
    if (!fetcher) return;
    let cancelled = false;
    setLoading(true);
    setError(false);
    fetcher(sport, gameId)
      .then((r) => { if (!cancelled) setData(r); })
      .catch(() => { if (!cancelled) { setData(null); setError(true); } })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [fetcher, sport, gameId]);

  useEffect(() => load(), [load]);

  if (!fetcher) return null;
  return (
    <div className={className}>
      <ExplainerPanel data={data} loading={loading} error={error} onRetry={() => load()} />
    </div>
  );
}
