import { useEffect, useState } from "react";
import { api, type StandingsRow } from "../api/client";

const STATUS: Record<StandingsRow["playoff_status"], string> = {
  clinched: "Clinched",
  "play-in": "Play-in",
  eliminated: "Out",
  "in-hunt": "In the hunt",
};

export default function StandingsPanel() {
  const [rows, setRows] = useState<StandingsRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getHubStandings().then(setRows).catch(() => setError("Couldn't load standings."));
  }, []);

  if (error) return <p className="text-[var(--color-shotclock)]">{error}</p>;
  if (rows === null) return <p>Loading standings…</p>;
  if (rows.length === 0) return <p>No standings data cached yet.</p>;

  const conferences: Array<"East" | "West"> = ["East", "West"];

  return (
    <div className="grid gap-6 sm:grid-cols-2">
      {conferences.map((conference) => (
        <div key={conference}>
          <h3 className="mb-2 font-semibold">{conference}</h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-[var(--color-net-faint)]">
                <th>Seed</th>
                <th>Team</th>
                <th>W-L</th>
                <th>GB</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {rows
                .filter((row) => row.conference === conference)
                .sort((a, b) => a.seed - b.seed)
                .map((row) => (
                  <tr key={row.abbreviation}>
                    <td>{row.seed}</td>
                    <td>{row.abbreviation}</td>
                    <td>
                      {row.wins}-{row.losses}
                    </td>
                    <td>{row.games_back === 0 ? "—" : row.games_back.toFixed(1)}</td>
                    <td>{STATUS[row.playoff_status] ?? row.playoff_status}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}