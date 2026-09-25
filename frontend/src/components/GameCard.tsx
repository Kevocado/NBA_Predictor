import type { Game, Prediction } from "../api/client";

// Lead with whoever the model favours. Home wins a 50/50 tie, matching the
// `>= 0.5` rule GameDetailModal uses to judge the pick after the game.
export function favourite(p: Prediction, home: string, away: string): { team: string; prob: number } {
  return p.home_win_probability >= 0.5 ? { team: home, prob: p.home_win_probability } : { team: away, prob: 1 - p.home_win_probability };
}

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
          <div className="text-base font-semibold">
            {game.away_team} <span className="text-[var(--color-net-faint)] font-normal">at</span> {game.home_team}
          </div>
        </div>
        {game.completed ? (
          <div className="shrink-0 text-right">
            <div className="stat-display text-2xl leading-none">
              {game.away_pts} <span className="text-[var(--color-net-faint)]">–</span> {game.home_pts}
            </div>
            <div className="mt-1 text-xs text-[var(--color-net-faint)]">Final</div>
          </div>
        ) : game.prediction ? (
          <FavouritePick prediction={game.prediction} home={game.home_team} away={game.away_team} />
        ) : (
          <div className="text-xs text-[var(--color-net-dim)]">No pick yet</div>
        )}
      </div>
    </button>
  );
}

function FavouritePick({ prediction, home, away }: { prediction: Prediction; home: string; away: string }) {
  const fav = favourite(prediction, home, away);
  return (
    <div className="shrink-0 text-right">
      <div className="stat-display text-3xl leading-none text-[var(--color-hardwood-bright)]">{Math.round(fav.prob * 100)}%</div>
      <div className="mt-1 text-xs text-[var(--color-net-dim)]">{fav.team} to win</div>
    </div>
  );
}
