import { useEffect, useState, type ReactNode } from "react";
import { EmptyState, ErrorState, Skeleton } from "../predictor-ui";

/**
 * Loading, error (with retry) and empty states for a Data Hub table, so each panel is just its table.
 * `load` must be a stable function (the api methods are); `noun` reads in the copy: "team" -> "team data".
 */
export default function HubRows<T>({ load, noun, children }: { load: () => Promise<T[]>; noun: string; children: (rows: T[]) => ReactNode }) {
  const [rows, setRows] = useState<T[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let stale = false;
    setRows(null);
    setFailed(false);
    load().then((r) => { if (!stale) setRows(r); }).catch(() => { if (!stale) setFailed(true); });
    return () => { stale = true; };
  }, [load, attempt]);

  if (failed) return <ErrorState message={`We couldn't load ${noun} data. Check your connection and try again.`} onRetry={() => setAttempt((n) => n + 1)} />;
  if (rows === null) return <Skeleton label={`Loading ${noun} data…`} />;
  if (rows.length === 0) return <EmptyState message={`No ${noun} data cached yet.`} />;
  return <>{children(rows)}</>;
}
