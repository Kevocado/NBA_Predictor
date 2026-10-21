import { useEffect, useRef, useState } from "react";
import EarlySeasonNotice from "../components/EarlySeasonNotice";
import GameDetailModal from "../components/GameDetailModal";
import TeamLogo from "../components/TeamLogo";
import { ModelRecordLine } from "../components/ModelRecordLine";
import { api, type Game, type TrackRecord } from "../api/client";
import { EmptyState, ErrorState, MatchCard, RoundNavigator, Skeleton } from "../predictor-ui";
import { addDays, mondayOf } from "../lib/weeks";
import { dayHeading, nextUpIds, tipZones, toCardModel, weekLabel, weekTally } from "../lib/nightCards";

/** The official logos on a card's two sides, as everywhere else on the site; the code chip is the fallback inside TeamLogo. */
const withLogos = (model: ReturnType<typeof toCardModel>) => ({
  ...model,
  left: { ...model.left, badge: <TeamLogo team={model.left.code} size={36} ariaHidden /> },
  right: { ...model.right, badge: <TeamLogo team={model.right.code} size={36} ariaHidden /> },
});

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

/** "3 games · 2 of 3 right": what is inside a folded day, and what happened. */
function daySummary(dayGames: Game[]): string {
  const { hits, settled } = weekTally(dayGames);
  return settled > 0 ? `${plural(dayGames.length, "game")} · ${hits} of ${settled} right` : plural(dayGames.length, "game");
}

type SortMode = "tip-off" | "confidence";

const byTip = (a: Game, b: Game) => (a.tip_off ?? "").localeCompare(b.tip_off ?? "") || a.game_id.localeCompare(b.game_id);
const byConfidence = (a: Game, b: Game) => {
  const ap = a.prediction?.home_win_probability ?? 0.5;
  const bp = b.prediction?.home_win_probability ?? 0.5;
  const aConf = Math.abs(ap - 0.5);
  const bConf = Math.abs(bp - 0.5);
  return bConf - aConf || a.game_id.localeCompare(b.game_id);
};

export default function GamesPage() {
  // The week the API lands on (the one with the next games), for "Jump to current week".
  const [currentWeek, setCurrentWeek] = useState<string | null>(null);
  const [weekStart, setWeekStart] = useState<string | null>(null);
  const [games, setGames] = useState<Game[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [sort, setSort] = useState<SortMode>("tip-off");
  // The trust strip's source. Loading and failure both stay `null`, and the
  // line is not drawn at all: a games page must not invent a record.
  const [trackRecord, setTrackRecord] = useState<TrackRecord[] | null>(null);
  // A deep link (?game=<id> on "/" — the hub's teaser rows point here) names
  // one game. Read once, on arrival, and matched against the week's games
  // after each successful load: only an id this page actually shows is
  // "known", so an unknown or absent identifier opens nothing and the visitor
  // gets the normal list — never an error dialog, never a blank screen. A
  // ref rather than state on purpose: clearing it must not re-run the
  // week-load effect (that would refetch and blank the list the moment the
  // detail opened), and it is cleared on a match so closing the detail does
  // not reopen it on the next week navigation. A failed load leaves it set,
  // so Try again still honours it.
  const deepLinkedGameId = useRef<string | null>(
    new URLSearchParams(window.location.search).get("game"),
  );

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
    api.getTrackRecord().then(setTrackRecord).catch(() => setTrackRecord(null));
  }, []);

  useEffect(() => {
    if (weekStart === null) return;
    setGames(null);
    setError(null);
    api
      .getGamesWeek(weekStart)
      .then((fetched) => {
        setGames(fetched);
        const wanted = deepLinkedGameId.current;
        if (wanted) {
          const match = fetched.find((game) => game.game_id === wanted);
          if (match) {
            setSelectedGameId(match.game_id);
            deepLinkedGameId.current = null;
          }
        }
      })
      .catch(() => setError("games"));
  }, [weekStart, reloadKey]);

  const step = (days: number) => setWeekStart((w) => addDays(w ?? mondayOf(new Date()), days));

  const sortedGames = [...(games ?? [])].sort(sort === "confidence" ? byConfidence : byTip);
  const isCurrentWeek = weekStart !== null && weekStart === currentWeek;
  const nextUp = nextUpIds(sortedGames, isCurrentWeek);
  const zones = tipZones(sortedGames);

  const gamesByDay = new Map<string, Game[]>();
  for (const game of sortedGames) {
    const existing = gamesByDay.get(game.game_date) ?? [];
    existing.push(game);
    gamesByDay.set(game.game_date, existing);
  }
  const days = [...gamesByDay.entries()].sort(([a], [b]) => a.localeCompare(b));
  // A week is up to seven nights of cards; on a phone that is a dozen screens. Open the day
  // with the next games and fold the rest into one-line summaries a tap away.
  const focusDay = (days.find(([, gs]) => gs.some((g) => !(g.completed && g.home_pts != null))) ?? days[0])?.[0];

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

      <ModelRecordLine rows={trackRecord} className="mb-4" />

      <EarlySeasonNotice />

      {error && <ErrorState message="We couldn't load this week's games. Check your connection and try again." onRetry={() => setReloadKey((k) => k + 1)} />}
      {!error && games === null && <Skeleton label="Loading games…" />}
      {!error && games !== null && games.length === 0 && (
        <EmptyState message="No games this week." action={{ label: "Go to next week", onClick: () => step(7) }} />
      )}
      {zones && <p className="mb-4 text-xs text-pr-text-dim">Tip-off times in {zones}</p>}

      <div className="mb-4 flex items-center gap-2 text-sm text-pr-text-dim">
        <span>Sort:</span>
        <button
          onClick={() => setSort("tip-off")}
          className={`px-2 py-1 rounded border text-xs ${sort === "tip-off" ? "bg-pr-accent text-pr-accent-ink" : "border-pr-rule hover:border-pr-accent"}`}
        >
          Tip-off order
        </button>
        <button
          onClick={() => setSort("confidence")}
          className={`px-2 py-1 rounded border text-xs ${sort === "confidence" ? "bg-pr-accent text-pr-accent-ink" : "border-pr-rule hover:border-pr-accent"}`}
        >
          Most confident
        </button>
      </div>

      <div className="space-y-2">
        {days.map(([day, dayGames]) => (
          <section key={day} aria-labelledby={`day-${day}`}>
            <details open={day === focusDay} className="group">
              <summary className="flex min-h-11 cursor-pointer list-none items-center justify-between gap-3 border-b border-pr-rule py-2 [&::-webkit-details-marker]:hidden">
                <h3 id={`day-${day}`} className="font-pr-display text-base font-semibold uppercase tracking-wide text-pr-text-dim">
                  {dayHeading(day)}
                </h3>
                <span className="flex items-center gap-2 text-xs text-pr-text-faint">
                  {daySummary(dayGames)}
                  <span aria-hidden="true" className="inline-block h-2 w-2 -rotate-45 border-b-2 border-r-2 border-current transition-transform group-open:rotate-45" />
                </span>
              </summary>
              <ul className="grid gap-3 pt-3 pb-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                {dayGames.map((game) => (
                  <li key={game.game_id} data-testid={`game-card-${game.game_id}`}>
                    <MatchCard {...withLogos(toCardModel(game, nextUp.has(game.game_id)))} moment="tip-off" compact onOpen={() => setSelectedGameId(game.game_id)} />
                  </li>
                ))}
              </ul>
            </details>
          </section>
        ))}
      </div>

      {selectedGameId && <GameDetailModal gameId={selectedGameId} onClose={() => setSelectedGameId(null)} />}
    </div>
  );
}
