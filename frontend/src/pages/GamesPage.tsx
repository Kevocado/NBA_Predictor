import { useEffect, useState } from "react";
import GameDetailModal from "../components/GameDetailModal";
import { api, type Game } from "../api/client";
import { EmptyState, ErrorState, MatchCard, RoundNavigator, Skeleton } from "../predictor-ui";
import { dayHeading, nextUpIds, tipZones, toCardModel, weekLabel, weekTally } from "../lib/nightCards";

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

const byTip = (a: Game, b: Game) => (a.tip_off ?? "").localeCompare(b.tip_off ?? "") || a.game_id.localeCompare(b.game_id);

export default function GamesPage() {
  // The week the API lands on (the one with the next games), for "Jump to current week".
  const [currentWeek, setCurrentWeek] = useState<string | null>(null);
  const [weekStart, setWeekStart] = useState<string | null>(null);
  const [games, setGames] = useState<Game[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    const land = (week: string) => {
      setCurrentWeek(week);
      setWeekStart(week);
    };
    api
      .getSeasonFirstWeek()
      .then((bounds) => land(bounds.first_week_start ?? mondayOf(new Date())))
      .catch(() => land(mondayOf(new Date())));
  }, []);

  useEffect(() => {
    if (weekStart === null) return;
    setGames(null);
    setError(null);
    api
      .getGamesWeek(weekStart)
      .then(setGames)
      .catch(() => setError("games"));
  }, [weekStart, reloadKey]);

  const step = (days: number) => setWeekStart((w) => addDays(w ?? mondayOf(new Date()), days));

  const gamesByDay = new Map<string, Game[]>();
  for (const game of [...(games ?? [])].sort(byTip)) {
    const existing = gamesByDay.get(game.game_date) ?? [];
    existing.push(game);
    gamesByDay.set(game.game_date, existing);
  }
  const days = [...gamesByDay.entries()].sort(([a], [b]) => a.localeCompare(b));
  const isCurrentWeek = weekStart !== null && weekStart === currentWeek;
  const nextUp = nextUpIds(games ?? [], isCurrentWeek);
  const zones = tipZones(games ?? []);

  return (
    <div>
      <RoundNavigator
        label={weekStart ? weekLabel(weekStart) : "This week"}
        unit="week"
        moment="tip-off"
        canPrev={weekStart !== null}
        canNext={weekStart !== null}
        onPrev={() => step(-7)}
        onNext={() => step(7)}
        onJumpToCurrent={!isCurrentWeek && currentWeek ? () => setWeekStart(currentWeek) : undefined}
        record={games ? weekTally(games) : undefined}
      />

      {error && <ErrorState message="We couldn't load this week's games. Check your connection and try again." onRetry={() => setReloadKey((k) => k + 1)} />}
      {!error && games === null && <Skeleton label="Loading games…" />}
      {!error && games !== null && games.length === 0 && (
        <EmptyState message="No games this week." action={{ label: "Go to next week", onClick: () => step(7) }} />
      )}
      {zones && <p className="mb-4 text-xs text-pr-text-dim">Tip-off times in {zones}</p>}

      <div className="space-y-6">
        {days.map(([day, dayGames]) => (
          <section key={day} aria-labelledby={`day-${day}`}>
            <h3 id={`day-${day}`} className="mb-2 font-pr-display text-base font-semibold uppercase tracking-wide text-pr-text-dim">
              {dayHeading(day)}
            </h3>
            <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {dayGames.map((game) => (
                <li key={game.game_id} data-testid={`game-card-${game.game_id}`}>
                  <MatchCard {...toCardModel(game, nextUp.has(game.game_id))} moment="tip-off" compact onOpen={() => setSelectedGameId(game.game_id)} />
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>

      {selectedGameId && <GameDetailModal gameId={selectedGameId} onClose={() => setSelectedGameId(null)} />}
    </div>
  );
}
