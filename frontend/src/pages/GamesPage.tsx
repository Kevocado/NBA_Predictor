import { useEffect, useState } from "react";
import GameCard from "../components/GameCard";
import GameDetailModal from "../components/GameDetailModal";
import { api, type Game } from "../api/client";

function toISODate(d: Date): string {
  return d.toISOString().slice(0, 10);
}

function mondayOf(d: Date): string {
  const day = d.getDay(); // 0 = Sunday, 1 = Monday, ...
  const diffToMonday = day === 0 ? -6 : 1 - day;
  const monday = new Date(d);
  monday.setDate(d.getDate() + diffToMonday);
  return toISODate(monday);
}

function addDays(isoDate: string, days: number): string {
  const d = new Date(isoDate + "T00:00:00");
  d.setDate(d.getDate() + days);
  return toISODate(d);
}

function formatWeekRange(weekStart: string): string {
  const start = new Date(weekStart + "T00:00:00");
  const end = new Date(addDays(weekStart, 6) + "T00:00:00");
  const fmt = (d: Date) => d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  return `${fmt(start)} – ${fmt(end)}`;
}

function formatDayHeader(isoDate: string): string {
  const d = new Date(isoDate + "T00:00:00");
  return d.toLocaleDateString(undefined, { weekday: "long", month: "short", day: "numeric" });
}

export default function GamesPage() {
  const [weekStart, setWeekStart] = useState(mondayOf(new Date()));
  const [games, setGames] = useState<Game[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null);

  useEffect(() => {
    setGames(null);
    setError(null);
    api
      .getGamesWeek(weekStart)
      .then(setGames)
      .catch(() => setError("Couldn't load games."));
  }, [weekStart]);

  const gamesByDay = new Map<string, Game[]>();
  for (const game of games ?? []) {
    const existing = gamesByDay.get(game.game_date) ?? [];
    existing.push(game);
    gamesByDay.set(game.game_date, existing);
  }

  return (
    <div>
      <div className="mb-6 flex items-center justify-between">
        <button
          aria-label="Previous week"
          onClick={() => setWeekStart((w) => addDays(w, -7))}
          className="text-[var(--color-net-dim)] hover:text-[var(--color-net)]"
        >
          ←
        </button>
        <span className="stat-display text-lg">{formatWeekRange(weekStart)}</span>
        <button
          aria-label="Next week"
          onClick={() => setWeekStart((w) => addDays(w, 7))}
          className="text-[var(--color-net-dim)] hover:text-[var(--color-net)]"
        >
          →
        </button>
      </div>

      {error && <p className="text-[var(--color-shotclock)]">{error}</p>}
      {!error && games === null && <p>Loading games…</p>}
      {!error && games !== null && games.length === 0 && <p>No games scheduled this week.</p>}

      <div className="space-y-6">
        {Array.from(gamesByDay.entries()).map(([day, dayGames]) => (
          <div key={day}>
            <h3 className="mb-2 text-sm text-[var(--color-net-faint)]">{formatDayHeader(day)}</h3>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {dayGames.map((game) => (
                <GameCard key={game.game_id} game={game} onSelect={setSelectedGameId} />
              ))}
            </div>
          </div>
        ))}
      </div>

      {selectedGameId && (
        <GameDetailModal gameId={selectedGameId} onClose={() => setSelectedGameId(null)} />
      )}
    </div>
  );
}
