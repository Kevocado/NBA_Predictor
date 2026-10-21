import { useEffect, useRef, useState } from "react";
import { api, type PlayerHubRow, type PlayerNextGame, type PlayerDetail } from "../api/client";
import { STAT_LABELS, type StatKey } from "../lib/boxScore";
import PlayerAvatar from "./PlayerAvatar";
import TeamLogo from "./TeamLogo";

/**
 * One player's detail: their last ten games, their averages, their next game.
 *
 * The games are drawn per stat: the actual as a bar, the model's projection as
 * a tick, and the band it says it can miss by as the shaded strip around the
 * tick. A bar inside the band is the model being as wrong as it said it would
 * be; a bar outside it is the model being wrong by more than it promised.
 */
export function PlayerDetailPanel({
  player,
  nextGame,
  bands,
}: {
  player: PlayerHubRow;
  nextGame?: PlayerNextGame | null;
  bands?: Record<string, number> | null;
}) {
  const [detail, setDetail] = useState<PlayerDetail | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);

  // **The focus moves into the record when it opens.** A reader who opened a
  // player's record should be reading it, not the table they came from, and a
  // screen reader says nothing about a panel that opened below the button.
  useEffect(() => {
    rootRef.current?.focus();
  }, []);

  /**
   * Close the record.
   *
   * The shared table owns the open row, and the only way to close it is the
   * toggle it drew. This finds that toggle -- the button in the row above,
   * the one that says it is expanded -- presses it, and puts the focus back
   * where the reader left it.
   */
  const close = () => {
    const toggle = rootRef.current
      ?.closest("table")
      ?.querySelector<HTMLButtonElement>('button[aria-expanded="true"]');
    toggle?.click();
    toggle?.focus();
  };

  useEffect(() => {
    let stale = false;
    setDetail(null);
    api
      .getHubPlayer(player.player_id)
      .then((result) => {
        if (!stale) setDetail(result);
      })
      .catch(() => {
        if (!stale) setDetail(null);
      });
    return () => {
      stale = true;
    };
  }, [player.player_id]);

  const games = detail?.games ?? [];

  return (
    <div
      ref={rootRef}
      tabIndex={-1}
      className="grid gap-4 sm:grid-cols-[auto_1fr]"
      data-testid={`player-detail-${player.player_id}`}
    >
      <div className="flex items-start gap-3">
        <PlayerAvatar playerId={player.player_id} name={player.player_name} size={64} />
        <div>
          <div className="flex items-start justify-between gap-3">
            <div className="text-base font-semibold">{player.player_name}</div>
            <button
              type="button"
              onClick={close}
              data-testid="player-record-close"
              className="shrink-0 rounded border border-pr-rule px-2 py-1 text-xs text-pr-text-dim hover:border-pr-accent"
            >
              Close
            </button>
          </div>
          <div className="flex items-center gap-1 text-xs text-pr-text-dim">
            <TeamLogo team={player.team} size={20} ariaHidden />
            {player.team} · {player.position || "—"}
          </div>
          {detail?.status && (
            <div className="mt-1 text-xs" data-testid="player-status">
              <span className={detail.status.status === "Out" ? "text-pr-loss" : "text-pr-text-dim"}>
                {detail.status.status}
              </span>
              <span className="text-pr-text-faint">
                {" "}
                · {detail.status.source}
                {detail.status.dated ? `, ${detail.status.dated.slice(0, 10)}` : ""}
              </span>
            </div>
          )}
        </div>
      </div>

      <div>
        <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-pr-text-dim" data-testid="player-averages">
          <span>{player.points_per_game.toFixed(1)} ppg</span>
          <span>{player.rebounds_per_game.toFixed(1)} rpg</span>
          <span>{player.assists_per_game.toFixed(1)} apg</span>
          <span>{player.minutes_per_game.toFixed(1)} min</span>
          <span>{player.three_pt_pct ? `${(player.three_pt_pct * 100).toFixed(1)}% from three` : "—"}</span>
        </div>

        {nextGame && (
          <div className="mb-2 text-xs text-pr-text-dim" data-testid="player-next-game">
            Next: vs {nextGame.opponent}
            {nextGame.projections?.points !== null && nextGame.projections?.points !== undefined
              ? ` · ${nextGame.projections.points.toFixed(1)} pts`
              : " · no projection"}
            {nextGame.projections?.rebounds !== null && nextGame.projections?.rebounds !== undefined
              ? `, ${nextGame.projections.rebounds.toFixed(1)} reb`
              : ""}
            {nextGame.projections?.assists !== null && nextGame.projections?.assists !== undefined
              ? `, ${nextGame.projections.assists.toFixed(1)} ast`
              : ""}
          </div>
        )}

        {games.length === 0 ? (
          <p className="text-sm text-pr-text-faint">No recorded games yet.</p>
        ) : (
          <div className="grid gap-3 sm:grid-cols-4">
            {(["points", "rebounds", "assists", "threes"] as StatKey[]).map((stat) => (
              <StatChart key={stat} stat={stat} games={games} band={bands?.[stat]} />
            ))}
          </div>
        )}
        {games.length > 0 && (
          <p className="mt-1 text-xs text-pr-text-faint">
            The last {games.length} games, oldest first. The bar is what the player did, the tick is what the model
            said, and the shaded strip is the band it says it can miss by.
          </p>
        )}
      </div>
    </div>
  );
}

/** One stat's last ten games: the actual as a bar, the projection as a tick. */
function StatChart({
  stat,
  games,
  band,
}: {
  stat: StatKey;
  games: PlayerDetail["games"];
  band?: number;
}) {
  const values = games.flatMap((game) => [
    game.actual[stat] ?? 0,
    (game.projection[stat] ?? 0) + (band ?? 0),
    (game.projection[stat] ?? 0) - (band ?? 0),
  ]);
  const max = Math.max(1, ...values);
  const height = (value: number) => `${Math.max(0, Math.min(100, (value / max) * 100))}%`;

  return (
    <div data-testid={`player-chart-${stat}`}>
      <div className="mb-1 text-xs uppercase tracking-wide text-pr-text-dim">{STAT_LABELS[stat]}</div>
      <div className="flex h-12 items-end gap-0.5">
        {games.map((game) => {
          const actual = game.actual[stat];
          const projection = game.projection[stat];
          return (
            <div key={game.game_id} className="relative h-full flex-1" title={`${game.game_date}: ${actual ?? "—"}`}>
              {projection !== null && band !== undefined && (
                <div
                  className="absolute inset-x-0 bg-pr-panel-2"
                  style={{
                    bottom: height(projection - band),
                    height: `${Math.max(1, ((2 * band) / max) * 100)}%`,
                  }}
                />
              )}
              {actual !== null && actual !== undefined && (
                <div className="absolute inset-x-0 bottom-0 bg-pr-accent" style={{ height: height(actual) }} />
              )}
              {projection !== null && projection !== undefined && (
                <div
                  className="absolute inset-x-0 border-t border-pr-text"
                  style={{ bottom: height(projection) }}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default PlayerDetailPanel;
