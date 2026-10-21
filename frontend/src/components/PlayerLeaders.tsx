import type { PlayerHubRow } from "../api/client";
import { leadersOn } from "../lib/playerFilters";
import PlayerAvatar from "./PlayerAvatar";
import TeamLogo from "./TeamLogo";

/**
 * The leaders strip: the five best players on each of the four stats a reader
 * checks first -- points, rebounds, assists, threes.
 *
 * The figures are the same per-game averages the table below carries, so the
 * strip and the table can never disagree. A player with no figure for a stat
 * is not on that list: "nobody shot 3.0 threes a game" is a different claim
 * from "the threes leader shot 0.0".
 */
export function PlayerLeaders({ rows, avatarSize = 32 }: { rows: PlayerHubRow[]; avatarSize?: number }) {
  const groups = [
    { key: "points", label: "Points", value: (row: PlayerHubRow) => row.points_per_game },
    { key: "rebounds", label: "Rebounds", value: (row: PlayerHubRow) => row.rebounds_per_game },
    { key: "assists", label: "Assists", value: (row: PlayerHubRow) => row.assists_per_game },
    { key: "threes", label: "Threes", value: (row: PlayerHubRow) => row.threes_per_game ?? null },
  ];

  return (
    <div className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-4" data-testid="player-leaders">
      {groups.map((group) => {
        const leaders = leadersOn(rows, group.value);
        return (
          <div key={group.key} className="rounded-pr border border-pr-rule bg-pr-panel p-3">
            <div className="mb-1 text-xs uppercase tracking-wide text-pr-text-dim">{group.label}</div>
            {leaders.length === 0 ? (
              <div className="text-sm text-pr-text-faint">No data yet.</div>
            ) : (
              <ul className="flex flex-col gap-1">
                {leaders.map((player, index) => (
                  <li key={player.player_id} className="flex items-center gap-2">
                    <span className="w-3 shrink-0 text-xs tabular-nums text-pr-text-faint">{index + 1}</span>
                    <PlayerAvatar playerId={player.player_id} name={player.player_name} size={avatarSize} />
                    <span className="min-w-0 flex-1">
                      {/* The name wraps rather than being cut: an ellipsis that
                          leaves "Shai Gilgeous-A…" tells the reader nothing. */}
                      <span className="block text-sm font-semibold leading-tight">{player.player_name}</span>
                      <span className="flex items-center gap-1 text-xs text-pr-text-dim">
                        <TeamLogo team={player.team} size={20} ariaHidden />
                        {(group.value(player) ?? 0).toFixed(1)}
                      </span>
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default PlayerLeaders;
