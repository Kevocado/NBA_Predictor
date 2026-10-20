import { useState } from "react";
import { api, type PlayerHubRow } from "../api/client";
import HubRows from "./HubRows";
import TeamLogo from "./TeamLogo";

const PAGE_SIZE = 20;

export default function PlayerHubPanel() {
  return (
    <HubRows<PlayerHubRow> load={api.getHubPlayers} noun="player">
      {(rows) => <PlayerTable rows={rows} />}
    </HubRows>
  );
}

function PlayerTable({ rows }: { rows: PlayerHubRow[] }) {
  const [page, setPage] = useState(0);
  const sorted = [...rows].sort((a, b) => b.rating - a.rating);
  const pageRows = sorted.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const hasNext = (page + 1) * PAGE_SIZE < sorted.length;

  return (
    <div>
      <p className="mb-3 max-w-prose text-sm text-pr-text-dim">Per-game averages. Early in the season they rest on a handful of games, so read them loosely.</p>
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
                <td>
                  <span className="inline-flex items-center gap-1.5">
                    <TeamLogo team={row.team} size={18} />
                    {row.team}
                  </span>
                </td>
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
      <p className="mt-3 text-xs text-pr-text-faint">Players {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, sorted.length)} of {sorted.length}</p>
      <div className="mt-2 flex gap-2">
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