import { useEffect, useState } from "react";
import { api, type PlayerHubRow } from "../api/client";

const PAGE_SIZE = 20;

export default function PlayerHubPanel() {
  const [rows, setRows] = useState<PlayerHubRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);

  useEffect(() => {
    api.getHubPlayers().then(setRows).catch(() => setError("Couldn't load player data."));
  }, []);

  if (error) return <p className="text-[var(--color-shotclock)]">{error}</p>;
  if (rows === null) return <p>Loading players…</p>;
  if (rows.length === 0) return <p>No player data cached yet.</p>;

  const sorted = [...rows].sort((a, b) => b.rating - a.rating);
  const pageRows = sorted.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const hasNext = (page + 1) * PAGE_SIZE < sorted.length;

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-[var(--color-net-faint)]">
              <th>Player</th>
              <th>Team</th>
              <th>Pos</th>
              <th>Rating</th>
              <th>PPG</th>
              <th>RPG</th>
              <th>APG</th>
              <th>FG%</th>
              <th>3P%</th>
              <th>FT%</th>
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row) => (
              <tr key={row.player_id}>
                <td>{row.player_name}</td>
                <td>{row.team}</td>
                <td>{row.position}</td>
                <td>{row.rating.toFixed(1)}</td>
                <td>{row.points_per_game.toFixed(1)}</td>
                <td>{row.rebounds_per_game.toFixed(1)}</td>
                <td>{row.assists_per_game.toFixed(1)}</td>
                <td>{(row.fg_pct * 100).toFixed(1)}</td>
                <td>{(row.three_pt_pct * 100).toFixed(1)}</td>
                <td>{(row.ft_pct * 100).toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-3 flex gap-2">
        <button data-testid="player-hub-prev" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
          Previous
        </button>
        <button data-testid="player-hub-next" disabled={!hasNext} onClick={() => setPage((p) => p + 1)}>
          Next
        </button>
      </div>
    </div>
  );
}