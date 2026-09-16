import { useEffect, useState } from "react";
import GameCard from "../components/GameCard";
import GameDetailModal from "../components/GameDetailModal";
import { api, type Game } from "../api/client";

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

export default function GamesPage() {
  const [date, setDate] = useState(today());
  const [games, setGames] = useState<Game[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null);

  useEffect(() => {
    setGames(null);
    setError(null);
    api
      .getGames(date)
      .then(setGames)
      .catch(() => setError("Couldn't load games."));
  }, [date]);

  return (
    <div>
      <label className="mb-4 flex items-center gap-2 text-sm text-[var(--color-net-dim)]">
        Date
        <input
          type="date"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          className="rounded border border-[var(--color-line)] bg-[var(--color-court-900)] px-2 py-1"
        />
      </label>

      {error && <p className="text-[var(--color-shotclock)]">{error}</p>}
      {!error && games === null && <p>Loading games…</p>}
      {!error && games !== null && games.length === 0 && <p>No games scheduled for this date.</p>}

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {games?.map((game) => (
          <GameCard key={game.game_id} game={game} onSelect={setSelectedGameId} />
        ))}
      </div>

      {selectedGameId && (
        <GameDetailModal gameId={selectedGameId} onClose={() => setSelectedGameId(null)} />
      )}
    </div>
  );
}