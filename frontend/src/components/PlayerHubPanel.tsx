import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, type PlayerHubRow, type PlayerNextGame } from "../api/client";
import { StatTable, type Column } from "../predictor-ui";
import HubRows from "./HubRows";
import TeamLogo from "./TeamLogo";
import PlayerAvatar from "./PlayerAvatar";
import PlayerLeaders from "./PlayerLeaders";
import PlayerDetailPanel from "./PlayerDetailPanel";
import { POSITIONS, filterPlayers, type PlayerFilters, type Position } from "../lib/playerFilters";

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
  const [more, setMore] = useState(false);
  const [nextGames, setNextGames] = useState<Record<string, PlayerNextGame>>({});
  const [bands, setBands] = useState<Record<string, number> | null>(null);
  const narrow = useNarrow();
  // The filters live in the URL query so a link keeps its filters: opening a
  // page someone sent you shows the same team, position, search and minutes
  // filter they saw when they copied the link.
  const [params, setParams] = useSearchParams();
  const initial = useMemo<PlayerFilters>(() => ({
    search: params.get("search") ?? "",
    team: params.get("team") ?? "",
    position: (POSITIONS.includes(params.get("position") as Position)
      ? params.get("position")
      : "All") as Position,
    minMinutes: params.get("minMinutes") !== "false",
  }), [params]);
  const [filters, setFilters] = useState<PlayerFilters>(initial);

  // The next games and the bands are the page's extras; neither may stop the
  // table from drawing.
  useEffect(() => {
    let stale = false;
    api
      .getHubPlayersNextGames()
      .then((result) => {
        if (!stale) setNextGames(result ?? {});
      })
      .catch(() => {
        if (!stale) setNextGames({});
      });
    api
      .getPlayerPropsManifest()
      .then((manifest) => {
        if (stale) return;
        const metrics = manifest?.metrics ?? {};
        const out: Record<string, number> = {};
        for (const [stat, figures] of Object.entries(metrics)) {
          const mae = (figures as { mae?: number | null })?.mae;
          if (typeof mae === "number") out[stat] = mae;
        }
        setBands(Object.keys(out).length ? out : null);
      })
      .catch(() => {
        if (!stale) setBands(null);
      });
    return () => {
      stale = true;
    };
  }, []);

  const teams = [...new Set(rows.map((r) => r.team))].sort();
  const filtered = filterPlayers(rows, filters);
  const sorted = [...filtered].sort((a, b) => b.rating - a.rating);
  const pageRows = sorted.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const hasNext = (page + 1) * PAGE_SIZE < sorted.length;
  const pageCount = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const showExtra = !narrow || more;

  const columns: Column<PlayerHubRow>[] = [
    {
      key: "player",
      label: "Player",
      value: (r) => r.player_name,
      render: (r) => (
        <span className="inline-flex items-center gap-2">
          <PlayerAvatar playerId={r.player_id} name={r.player_name} size={32} />
          {r.player_name}
        </span>
      ),
    },
    {
      key: "team",
      label: "Team",
      value: (r) => r.team,
      render: (r) => (
        <span className="inline-flex items-center gap-1.5">
          <TeamLogo team={r.team} size={20} />
          {r.team}
        </span>
      ),
    },
    { key: "ppg", label: "PPG", value: (r) => r.points_per_game, numeric: true, render: (r) => r.points_per_game.toFixed(1) },
    { key: "rpg", label: "RPG", value: (r) => r.rebounds_per_game, numeric: true, render: (r) => r.rebounds_per_game.toFixed(1) },
    { key: "apg", label: "APG", value: (r) => r.assists_per_game, numeric: true, render: (r) => r.assists_per_game.toFixed(1) },
    ...(showExtra
      ? ([
          { key: "pos", label: "Pos", value: (r) => r.position },
          { key: "rating", label: "Rating", value: (r) => r.rating, numeric: true, render: (r) => r.rating.toFixed(1) },
          { key: "min", label: "Min", value: (r) => r.minutes_per_game, numeric: true, render: (r) => r.minutes_per_game.toFixed(1) },
          { key: "fg", label: "FG%", value: (r) => r.fg_pct, numeric: true, render: (r) => (r.fg_pct * 100).toFixed(1) },
          { key: "three", label: "3P%", value: (r) => r.three_pt_pct, numeric: true, render: (r) => (r.three_pt_pct * 100).toFixed(1) },
          { key: "ft", label: "FT%", value: (r) => r.ft_pct, numeric: true, render: (r) => (r.ft_pct * 100).toFixed(1) },
        ] satisfies Column<PlayerHubRow>[])
      : []),
    {
      key: "next",
      label: "Next",
      tooltip: "The next game, and what the model expects: points / rebounds / assists",
      value: (r) => nextGames[r.player_id]?.game_date ?? null,
      render: (r) => {
        const next = nextGames[r.player_id];
        if (!next) return <span className="text-pr-text-faint">—</span>;
        return (
          <span
            className="inline-flex items-center gap-1.5"
            data-testid="player-next"
            title="Projected points / rebounds / assists"
          >
            <TeamLogo team={next.opponent} size={20} ariaHidden />
            <span className="tabular-nums">
              {next.projections?.points !== null && next.projections?.points !== undefined
                ? next.projections.points.toFixed(1)
                : "—"}
              {next.projections?.rebounds !== null && next.projections?.rebounds !== undefined
                ? ` / ${next.projections.rebounds.toFixed(1)}`
                : ""}
              {next.projections?.assists !== null && next.projections?.assists !== undefined
                ? ` / ${next.projections.assists.toFixed(1)}`
                : ""}
            </span>
          </span>
        );
      },
    },
  ];

  const setFilter = (patch: Partial<PlayerFilters>) => {
    setFilters((current) => {
      const next = { ...current, ...patch };
      const q = new URLSearchParams(params);
      if (next.search) q.set("search", next.search); else q.delete("search");
      if (next.team) q.set("team", next.team); else q.delete("team");
      if (next.position !== "All") q.set("position", next.position); else q.delete("position");
      if (!next.minMinutes) q.set("minMinutes", "false"); else q.delete("minMinutes");
      setParams(q, { replace: true });
      return next;
    });
    setPage(0);
  };

  return (
    <div>
      <p className="mb-3 max-w-prose text-sm text-pr-text-dim">Per-game averages. Early in the season they rest on a handful of games, so read them loosely.</p>

      <PlayerLeaders rows={filtered} avatarSize={narrow ? 28 : 32} />

      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <label htmlFor="player-hub-search" className="text-pr-text-dim">
          Search
        </label>
        <input
          id="player-hub-search"
          type="search"
          value={filters.search}
          onChange={(e) => setFilter({ search: e.target.value })}
          placeholder="Name or team"
          className="rounded border border-pr-rule bg-pr-panel px-2 py-1 text-sm"
        />
        <label className={!narrow ? "text-pr-text-dim" : "sr-only"} htmlFor="player-hub-team">
          Team
        </label>
        {narrow ? (
          <div className="inline-flex rounded border border-pr-rule gap-1" role="group" aria-label="Team">
            <button
              type="button"
              aria-pressed={filters.team === ""}
              onClick={() => setFilter({ team: "" })}
              className={`px-2 py-1 text-xs ${filters.team === "" ? "bg-pr-accent font-semibold text-pr-accent-ink" : "text-pr-text-dim"}`}
            >
              All teams
            </button>
            {teams.map((t) => (
              <button
                key={t}
                type="button"
                aria-pressed={filters.team === t}
                onClick={() => setFilter({ team: t })}
                className={`px-2 py-1 text-xs ${filters.team === t ? "bg-pr-accent font-semibold text-pr-accent-ink" : "text-pr-text-dim"}`}
              >
                {t}
              </button>
            ))}
          </div>
        ) : (
          <select
            id="player-hub-team"
            value={filters.team}
            onChange={(e) => setFilter({ team: e.target.value })}
            className="rounded border border-pr-rule bg-pr-panel px-2 py-1 text-sm"
          >
            <option value="">All teams</option>
            {teams.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        )}
        <div className="inline-flex rounded border border-pr-rule" role="group" aria-label="Position">
          {POSITIONS.map((position: Position) => (
            <button
              key={position}
              type="button"
              aria-pressed={filters.position === position}
              onClick={() => setFilter({ position })}
              className={`px-2 py-1 text-xs ${filters.position === position ? "bg-pr-accent font-semibold text-pr-accent-ink" : "text-pr-text-dim"}`}
            >
              {position}
            </button>
          ))}
        </div>
        <label className="inline-flex items-center gap-1.5 text-pr-text-dim">
          <input
            type="checkbox"
            checked={filters.minMinutes}
            onChange={(e) => setFilter({ minMinutes: e.target.checked })}
          />
          Min 15 minutes
        </label>
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
          expand={(row) => (
            <PlayerDetailPanel player={row} nextGame={nextGames[row.player_id]} bands={bands} />
          )}
        />
      </div>

      <p className="mt-3 text-xs text-pr-text-faint">
        Players {sorted.length === 0 ? 0 : page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, sorted.length)} of {sorted.length}
        {sorted.length !== rows.length ? ` (of ${rows.length})` : ""}
      </p>
      {/* The pager: 44px targets, because a phone is read with a thumb, and the
          page number in words, because "Page 1 of 3" says where you are. */}
      <div className="mt-2 flex items-center gap-3">
        <button
          data-testid="player-hub-prev"
          disabled={page === 0}
          onClick={() => setPage((p) => p - 1)}
          className="min-h-11 min-w-11 rounded border border-pr-rule px-3 text-sm text-pr-text disabled:opacity-40"
        >
          Previous
        </button>
        <span className="text-sm text-pr-text-dim" data-testid="player-hub-page">
          Page {page + 1} of {pageCount}
        </span>
        <button
          data-testid="player-hub-next"
          disabled={!hasNext}
          onClick={() => setPage((p) => p + 1)}
          className="min-h-11 min-w-11 rounded border border-pr-rule px-3 text-sm text-pr-text disabled:opacity-40"
        >
          Next
        </button>
      </div>
    </div>
  );
}
