import { useEffect, useState } from "react";
import { api, type PlayerHubRow } from "../api/client";
import { StatTable, type Column } from "../predictor-ui";
import HubRows from "./HubRows";
import TeamLogo from "./TeamLogo";

const PAGE_SIZE = 20;

/** The narrow layout is a column set, not a second table: one StatTable keeps
 *  the sort when the extra stats are revealed. */
function useNarrow() {
  const query = "(max-width: 639px)";
  const [narrow, setNarrow] = useState(
    () => typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia(query).matches,
  );
  useEffect(() => {
    const mq = typeof window.matchMedia === "function" ? window.matchMedia(query) : null;
    if (!mq) return;
    const sync = () => setNarrow(mq.matches);
    sync();
    mq.addEventListener?.("change", sync);
    return () => mq.removeEventListener?.("change", sync);
  }, []);
  return narrow;
}

export default function PlayerHubPanel() {
  return (
    <HubRows<PlayerHubRow> load={api.getHubPlayers} noun="player">
      {(rows) => <PlayerTable rows={rows} />}
    </HubRows>
  );
}

function PlayerTable({ rows }: { rows: PlayerHubRow[] }) {
  const [page, setPage] = useState(0);
  const [team, setTeam] = useState("");
  const [more, setMore] = useState(false);
  const narrow = useNarrow();

  const teams = [...new Set(rows.map((r) => r.team))].sort();
  const filtered = team ? rows.filter((r) => r.team === team) : rows;
  const sorted = [...filtered].sort((a, b) => b.rating - a.rating);
  const pageRows = sorted.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const hasNext = (page + 1) * PAGE_SIZE < sorted.length;
  const showExtra = !narrow || more;

  const columns: Column<PlayerHubRow>[] = [
    { key: "player", label: "Player", value: (r) => r.player_name },
    {
      key: "team",
      label: "Team",
      value: (r) => r.team,
      render: (r) => (
        <span className="inline-flex items-center gap-1.5">
          <TeamLogo team={r.team} size={18} />
          {r.team}
        </span>
      ),
    },
    { key: "pos", label: "Pos", value: (r) => r.position },
    { key: "ppg", label: "PPG", value: (r) => r.points_per_game, numeric: true, render: (r) => r.points_per_game.toFixed(1) },
    { key: "rpg", label: "RPG", value: (r) => r.rebounds_per_game, numeric: true, render: (r) => r.rebounds_per_game.toFixed(1) },
    { key: "apg", label: "APG", value: (r) => r.assists_per_game, numeric: true, render: (r) => r.assists_per_game.toFixed(1) },
    ...(showExtra
      ? ([
          { key: "rating", label: "Rating", value: (r) => r.rating, numeric: true, render: (r) => r.rating.toFixed(1) },
          { key: "fg", label: "FG%", value: (r) => r.fg_pct, numeric: true, render: (r) => (r.fg_pct * 100).toFixed(1) },
          { key: "three", label: "3P%", value: (r) => r.three_pt_pct, numeric: true, render: (r) => (r.three_pt_pct * 100).toFixed(1) },
          { key: "ft", label: "FT%", value: (r) => r.ft_pct, numeric: true, render: (r) => (r.ft_pct * 100).toFixed(1) },
        ] satisfies Column<PlayerHubRow>[])
      : []),
  ];

  return (
    <div>
      <p className="mb-3 max-w-prose text-sm text-pr-text-dim">Per-game averages. Early in the season they rest on a handful of games, so read them loosely.</p>

      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <label htmlFor="player-hub-team" className="text-pr-text-dim">
          Team
        </label>
        <select
          id="player-hub-team"
          value={team}
          onChange={(e) => {
            setTeam(e.target.value);
            setPage(0);
          }}
          className="rounded border border-pr-rule bg-pr-panel px-2 py-1 text-sm"
        >
          <option value="">All teams</option>
          {teams.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
        {narrow && (
          <button
            type="button"
            aria-expanded={more}
            onClick={() => setMore((v) => !v)}
            className="rounded border border-pr-rule px-2 py-1 text-xs hover:border-pr-accent"
          >
            {more ? "Fewer stats" : "More stats"}
          </button>
        )}
      </div>

      <div className={narrow ? "player-table-compact" : undefined}>
        <StatTable
          rows={pageRows}
          columns={columns}
          rowKey={(r) => r.player_id}
          initialSort={{ key: "rating", dir: "desc" }}
          caption="NBA players, per-game averages"
        />
      </div>

      <p className="mt-3 text-xs text-pr-text-faint">
        Players {sorted.length === 0 ? 0 : page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, sorted.length)} of {sorted.length}
      </p>
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
