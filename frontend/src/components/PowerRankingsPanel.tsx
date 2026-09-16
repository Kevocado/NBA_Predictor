import { useEffect, useState } from "react";
import { api, type PowerRankingRow } from "../api/client";

const TREND_GLYPH: Record<PowerRankingRow["trend"], string> = { up: "▲", down: "▼", steady: "–" };
const TREND_COLOR: Record<PowerRankingRow["trend"], string> = {
  up: "text-[var(--color-win)]",
  down: "text-[var(--color-shotclock)]",
  steady: "text-[var(--color-net-faint)]",
};

export default function PowerRankingsPanel() {
  const [rows, setRows] = useState<PowerRankingRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getHubRankings().then(setRows).catch(() => setError("Couldn't load rankings."));
  }, []);

  if (error) return <p className="text-[var(--color-shotclock)]">{error}</p>;
  if (rows === null) return <p>Loading rankings…</p>;
  if (rows.length === 0) return <p>No ranking data cached yet.</p>;

  return (
    <ol className="space-y-1 text-sm">
      {[...rows]
        .sort((a, b) => a.rank - b.rank)
        .map((row) => (
          <li key={row.abbreviation} className="flex justify-between border-b border-[var(--color-line)] py-1">
            <span>
              #{row.rank} <span>{row.abbreviation}</span>
            </span>
            <span>
              {row.power_rating.toFixed(0)} <span className={TREND_COLOR[row.trend]}>{TREND_GLYPH[row.trend]}</span>
            </span>
          </li>
        ))}
    </ol>
  );
}