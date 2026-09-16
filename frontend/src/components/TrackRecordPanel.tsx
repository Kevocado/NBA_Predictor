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
        {rows.map((row) => (
          <tr key={row.market}>
            <td>{row.market}</td>
            <td>{row.total_predictions}</td>
            <td>{row.correct_predictions}</td>
            <td>{Math.round(row.hit_rate * 100)}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}