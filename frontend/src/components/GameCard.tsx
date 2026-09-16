import type { Game } from "../api/client";

interface GameCardProps {
  game: Game;
  onSelect: (gameId: string) => void;
}

export default function GameCard({ game, onSelect }: GameCardProps) {
  return (
    <button
      data-testid={`game-card-${game.game_id}`}
      onClick={() => onSelect(game.game_id)}
      className="w-full rounded border border-[var(--color-line)] bg-[var(--color-court-900)] p-4 text-left transition hover:border-[var(--color-hardwood)]"
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-xs text-[var(--color-net-faint)]">{game.game_date}</div>
          <div className="mt-1 text-base font-semibold">
            {game.away_team} <span className="text-[var(--color-net-faint)] font-normal">at</span> {game.home_team}
          </div>
        </div>
        {game.prediction ? (
          <div className="shrink-0 text-right">
            <div className="stat-display text-3xl leading-none text-[var(--color-hardwood-bright)]">
              {Math.round(game.prediction.home_win_probability * 100)}%
            </div>
            <div className="mt-1 text-xs text-[var(--color-net-faint)]">{game.home_team} to win</div>
          </div>
        ) : (
          <div className="text-xs text-[var(--color-net-dim)]">Pending</div>
        )}
      </div>
    </button>
  );
}
