import { useEffect, useState } from "react";
import { api, type TrackRecord } from "../api/client";

export default function TrackRecordPanel() {
  const [rows, setRows] = useState<TrackRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getTrackRecord().then(setRows).catch(() => setError("Couldn't load track record."));
  }, []);

  if (error) return <p className="text-[var(--color-shotclock)]">{error}</p>;
  if (rows === null) return <p>Loading track record…</p>;
  if (rows.length === 0) return <p>No tracked predictions yet.</p>;

  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-left text-[var(--color-net-faint)]">
          <th>Market</th>
          <th>Predictions</th>
          <th>Correct</th>
          <th>Hit Rate</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const pct = Math.round(row.hit_rate * 100);
          return (
            <tr key={row.market}>
              <td>{row.market}</td>
              <td>{row.total_predictions}</td>
              <td>{row.correct_predictions}</td>
              {row.market === "player_props" ? (
                <td title="Hit rate doesn't apply to continuous prop errors — see MAE instead">—</td>
              ) : (
                <td>
                  <div className="flex items-center gap-2">
                    <div
                      className="relative h-2 w-24 rounded bg-[var(--color-net)]"
                      role="img"
                      aria-label={`${row.market} hit rate ${pct} percent`}
                    >
                      <div
                        className="h-full rounded bg-[var(--color-win)]"
                        style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
                      />
                      <div
                        data-testid="break-even-50"
                        title="Break-even (50%)"
                        aria-hidden="true"
                        className="absolute inset-y-0 w-px -translate-x-1/2 bg-[var(--color-shotclock)]"
                        style={{ left: "50%" }}
                      />
                    </div>
                    <span>{pct}%</span>
                    <span className="text-xs text-[var(--color-net-faint)]" title="Break-even (50%)">
                      50%
                    </span>
                  </div>
                </td>
              )}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}