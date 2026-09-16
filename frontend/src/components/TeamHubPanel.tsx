import { useEffect, useState } from "react";
import { api, type TeamHubRow } from "../api/client";

export default function TeamHubPanel() {
  const [rows, setRows] = useState<TeamHubRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getHubTeams().then(setRows).catch(() => setError("Couldn't load team data."));
  }, []);

  if (error) return <p className="text-[var(--color-shotclock)]">{error}</p>;
  if (rows === null) return <p>Loading teams…</p>;
  if (rows.length === 0) return <p>No team data cached yet.</p>;

  const conferences: Array<"East" | "West"> = ["East", "West"];

  return (
    <div className="space-y-6">
      {conferences.map((conference) => (
        <div key={conference}>
          <h3 className="mb-2 font-semibold">{conference}</h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-[var(--color-net-faint)]">
                <th>Team</th>
                <th>W-L</th>
                <th>PPG</th>
                <th>Opp PPG</th>
                <th>Net Rtg</th>
                <th>Pace</th>
                <th>Streak</th>
              </tr>
            </thead>
            <tbody>
              {rows
                .filter((row) => row.conference === conference)
                .sort((a, b) => b.points_per_game - a.points_per_game)
                .map((row) => (
                  <tr key={row.abbreviation}>
                    <td>{row.abbreviation}</td>
                    <td>
                      {row.wins}-{row.losses}
                    </td>
                    <td>{row.points_per_game.toFixed(1)}</td>
                    <td>{row.opp_points_per_game.toFixed(1)}</td>
                    <td>{row.net_rating.toFixed(1)}</td>
                    <td>{row.pace.toFixed(1)}</td>
                    <td>{row.streak}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}