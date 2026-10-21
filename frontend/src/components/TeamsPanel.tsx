import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  api,
  type FormEntry,
  type PlayerHubRow,
  type PowerRankingRow,
  type StandingsRow,
  type TeamHubRow,
  type TeamOverview,
} from "../api/client";
import { StatTable, type Column } from "../predictor-ui";
import { rankTier } from "../predictor-ui/lib/rankTier";
import { RatingList, RatingValue } from "./ratings";
import { standingsStatus } from "../lib/seasonCaption";
import { teamName } from "../lib/teams";
import TeamLogo from "./TeamLogo";

/**
 * The Teams page: every team, where it stands, and what it is like.
 *
 * One table, in the standings format, because a reader asking "where does my
 * team stand" wants one table and not three: the standings, the power
 * rankings and the ratings were three tabs saying three things about the same
 * thirty teams.
 *
 * Every figure is the server's. The standings and the power ratings are the
 * caches the ingest writes; the ratings come from the same box-score window
 * the game detail reads, so a team's rating here and its rating on a game
 * page are the same number.
 */

const STATUS_LABEL: Record<string, string> = {
  clinched: "Leading",
  "play-in": "Play-in",
  eliminated: "Trailing",
  "in-hunt": "In the hunt",
};

type Conference = "East" | "West" | "League";

interface Row {
  team: string;
  seed: number;
  wins: number;
  losses: number;
  gamesBack: number;
  status: string;
  power: number | null;
  powerRank: number | null;
  trend: string;
  net: { value: number | null; rank: number | null };
  off: { value: number | null; rank: number | null };
  def: { value: number | null; rank: number | null };
  pace: { value: number | null; rank: number | null };
  form: FormEntry[];
  streak: number;
}

export default function TeamsPanel() {
  // The old power-rankings link lands here with ?sort=power, and it still means
  // "by power rating" when it does.
  const [params] = useSearchParams();
  const [data, setData] = useState<
    | {
        standings: StandingsRow[];
        rankings: PowerRankingRow[];
        teams: TeamHubRow[];
        overview: TeamOverview;
        players: PlayerHubRow[];
      }
    | null
  >(null);
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [conference, setConference] = useState<Conference>("League");

  useEffect(() => {
    let stale = false;
    setData(null);
    setFailed(false);
    Promise.all([
      api.getHubStandings(),
      api.getHubRankings(),
      api.getHubTeams(),
      api.getHubTeamOverview(),
      api.getHubPlayers(),
    ])
      .then(([standings, rankings, teams, overview, players]) => {
        if (!stale) setData({ standings, rankings, teams, overview, players });
      })
      .catch(() => {
        if (!stale) setFailed(true);
      });
    return () => {
      stale = true;
    };
  }, [attempt]);

  const rows = useMemo(() => (data ? buildRows(data) : []), [data]);
  // Until every team has played there is no standings order to show (a pile of unseeded 0s): lead with the power rating.
  const everyTeamSeeded = rows.length > 0 && rows.every((row) => row.seed > 0);
  const sortParam = params.get("sort") === "power" || !everyTeamSeeded ? "power" : "seed";

  if (failed) {
    return (
      <p className="text-sm text-pr-text-dim" data-testid="teams-error">
        We couldn't load the team data. Check your connection and try again.
      </p>
    );
  }
  if (!data) {
    return (
      <p className="text-sm text-pr-text-dim" data-testid="teams-loading">
        Loading team data…
      </p>
    );
  }
  if (rows.length === 0) {
    return <p className="text-sm text-pr-text-dim">No team data cached yet.</p>;
  }

  const shown = rows.filter((row) => conference === "League" || teamConference(data.standings, row.team) === conference);
  const mostGamesPlayed = Math.max(...data.standings.map((r) => r.wins + r.losses));
  const state = data.standings[0]?.season_state;
  const note =
    state === "regular" || state === "postseason"
      ? null
      : state === "offseason"
        ? "Off-season — last season's record. No playoff race yet."
        : "Pre-season — schedule loaded, nobody has played.";

  return (
    <div data-testid="teams-panel">
      <h2 className="mb-1 text-lg font-semibold">Teams</h2>
      {note && <p className="mb-3 text-sm text-pr-text-dim">{note}</p>}

      <LeaderTiles data={data} />

      <div className="mb-3 mt-4 flex flex-wrap items-center gap-2">
        <div className="inline-flex rounded-pr border border-pr-rule" role="group" aria-label="Conference">
          {(["East", "West", "League"] as Conference[]).map((name) => (
            <button
              key={name}
              type="button"
              onClick={() => setConference(name)}
              aria-pressed={conference === name}
              className={`px-3 py-1.5 text-sm ${conference === name ? "bg-pr-accent font-semibold text-pr-accent-ink" : "text-pr-text-dim"}`}
            >
              {name}
            </button>
          ))}
        </div>
        {data.overview.caption && (
          <span className="text-xs text-pr-text-faint">{data.overview.caption}</span>
        )}
      </div>

      <TeamsTable
        rows={shown}
        overview={data.overview}
        players={data.players}
        initialSortKey={sortParam}
      />
    </div>
  );
}

function teamConference(standings: StandingsRow[], team: string): string {
  return standings.find((row) => row.abbreviation === team)?.conference ?? "";
}

/** One row per team, from the four feeds joined on the team code. */
function buildRows(data: {
  standings: StandingsRow[];
  rankings: PowerRankingRow[];
  teams: TeamHubRow[];
  overview: TeamOverview;
}): Row[] {
  const byStanding = new Map(data.standings.map((row) => [row.abbreviation, row]));
  const byRanking = new Map(data.rankings.map((row) => [row.abbreviation, row]));
  const byTeam = new Map(data.teams.map((row) => [row.abbreviation, row]));
  const figure = (team: string, key: string) => data.overview.ratings?.[team]?.[key] ?? { value: null, rank: null };

  const teams = new Set<string>([
    ...data.standings.map((r) => r.abbreviation),
    ...data.teams.map((r) => r.abbreviation),
  ]);

  return [...teams].map((team) => {
    const standing = byStanding.get(team);
    const ranking = byRanking.get(team);
    const hub = byTeam.get(team);
    return {
      team,
      seed: standing?.seed ?? 0,
      wins: standing?.wins ?? 0,
      losses: standing?.losses ?? 0,
      gamesBack: standing?.games_back ?? 0,
      status: standing?.playoff_status ? STATUS_LABEL[standing.playoff_status] : "—",
      power: ranking?.power_rating ?? null,
      powerRank: ranking?.rank ?? null,
      trend: ranking?.trend ?? "steady",
      net: figure(team, "net"),
      off: figure(team, "ortg"),
      def: figure(team, "drtg"),
      pace: figure(team, "pace"),
      form: data.overview.form?.[team] ?? [],
      streak: hub?.streak ?? 0,
    };
  });
}

/** The last ten results, oldest first: a dot per game, green or red. */
/**
 * The last ten results, oldest first: a dot per game, green or red.
 *
 * **Last season's games are dimmed.** Early in a season a team has played one
 * or two games, so its last ten are mostly last season's -- and a row of
 * full-coloured dots reads as this season's form when it is not. The cell says
 * how many of them are from last season.
 */
function FormDots({ form }: { form: FormEntry[] }) {
  if (form.length === 0) return <span className="text-pr-text-faint">—</span>;
  const fromLastSeason = form.filter((entry) => !entry.this_season).length;
  const label = `${form.map((entry) => entry.result).join(" ")}${
    fromLastSeason > 0 ? ` — ${fromLastSeason} from last season` : ""
  }`;
  return (
    <span className="inline-flex items-center gap-0.5" aria-label={label} title={label}>
      {form.map((entry, i) => (
        <span
          key={i}
          data-last-season={entry.this_season ? undefined : "true"}
          className={`inline-block h-2 w-2 rounded-full ${
            entry.result === "W" ? "bg-pr-win" : "bg-pr-loss"
          } ${entry.this_season ? "" : "opacity-40"}`}
          aria-hidden="true"
        />
      ))}
    </span>
  );
}

/** The four tiles: the league's best at each thing. */
function LeaderTiles({ data }: { data: { overview: TeamOverview; rankings: PowerRankingRow[] } }) {
  const ratings = data.overview.ratings ?? {};
  const n = data.overview.n_teams || 30;

  // The ranks are already "first is best" for every metric -- a defence is
  // ranked on what it ALLOWS, so rank 1 is the team that concedes least.
  const best = (key: string, decimals = 1) => {
    for (const [team, figures] of Object.entries(ratings)) {
      const figure = figures[key];
      if (figure && figure.value !== null && figure.rank === 1) {
        return { team, value: figure.value.toFixed(decimals), rank: figure.rank, tier: rankTier(figure.rank, n) };
      }
    }
    return null;
  };

  const offence = best("ortg");
  const defence = best("drtg");
  const pace = best("pace");
  const power = data.rankings.length
    ? {
        team: data.rankings[0].abbreviation,
        value: data.rankings[0].power_rating.toFixed(1),
        rank: data.rankings[0].rank,
        tier: rankTier(data.rankings[0].rank, 30),
      }
    : null;

  const tiles = [
    { title: "Best offence", tile: offence, help: "Points scored per 100 possessions." },
    { title: "Best defence", tile: defence, help: "Points allowed per 100 possessions. Fewer is better." },
    { title: "Fastest pace", tile: pace, help: "Possessions per game. More possessions means more shots." },
    { title: "Highest power rating", tile: power, help: "The model's Elo rating." },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4" data-testid="leader-tiles">
      {tiles.map(({ title, tile, help }) => (
        <div key={title} className="rounded-pr border border-pr-rule bg-pr-panel p-3">
          <div className="mb-1 text-xs uppercase tracking-wide text-pr-text-dim" title={help}>
            {title}
          </div>
          {tile ? (
            <div className="flex items-center gap-2">
              <TeamLogo team={tile.team} size={28} ariaHidden />
              <span className="min-w-0">
                <span className="block truncate text-sm font-semibold">{teamName(tile.team) || tile.team}</span>
                <span className="block text-xs text-pr-text-dim">
                  {tile.value}{" "}
                  <span className={`pr-rank-box pr-rank-${tile.tier} pr-rating-rank`}>{tile.rank}</span>
                </span>
              </span>
            </div>
          ) : (
            <div className="text-sm text-pr-text-faint">No data yet.</div>
          )}
        </div>
      ))}
    </div>
  );
}

/** The table: the standings format, sortable, with a row's detail under it. */
function TeamsTable({
  rows,
  overview,
  players,
  initialSortKey,
}: {
  rows: Row[];
  overview: TeamOverview;
  players: PlayerHubRow[];
  initialSortKey: string;
}) {
  const n = overview.n_teams || 30;

  const columns: Column<Row>[] = [
    {
      key: "team",
      label: "Team",
      value: (row) => teamName(row.team) || row.team,
      render: (row) => (
        <span className="inline-flex items-center gap-2">
          <TeamLogo team={row.team} size={28} ariaHidden />
          <span className="truncate">{teamName(row.team) || row.team}</span>
        </span>
      ),
    },
    { key: "seed", label: "Seed", value: (row) => row.seed, numeric: true, firstDir: "asc", render: (row) => (row.seed === 0 ? "—" : row.seed) },
    { key: "record", label: "W-L", value: (row) => `${row.wins}-${row.losses}` },
    {
      key: "gb",
      label: "GB",
      value: (row) => row.gamesBack,
      numeric: true,
      firstDir: "asc",
      render: (row) => (row.gamesBack === 0 ? "—" : row.gamesBack.toFixed(1)),
    },
    { key: "status", label: "Status", value: (row) => row.status },
    {
      key: "power",
      label: "Power",
      value: (row) => row.power,
      numeric: true,
      render: (row) => (
        <span className="inline-flex items-center justify-end gap-1.5">
          {row.trend === "up" ? "▲" : row.trend === "down" ? "▼" : "—"}
          <span className="tabular-nums">{row.power === null ? "—" : row.power.toFixed(1)}</span>
          {row.powerRank !== null && (
            <span className={`pr-rank-box pr-rank-${rankTier(row.powerRank, 30)} pr-rating-rank`}>{row.powerRank}</span>
          )}
        </span>
      ),
    },
    {
      key: "net",
      label: "Net",
      value: (row) => row.net.value,
      numeric: true,
      render: (row) => <RatingValue figure={row.net} n={n} label="net rating" />,
    },
    {
      key: "off",
      label: "Off",
      value: (row) => row.off.value,
      numeric: true,
      render: (row) => <RatingValue figure={row.off} n={n} label="offence" />,
    },
    {
      key: "def",
      label: "Def",
      value: (row) => row.def.value,
      numeric: true,
      render: (row) => <RatingValue figure={row.def} n={n} label="defence" />,
    },
    {
      key: "pace",
      label: "Pace",
      value: (row) => row.pace.value,
      numeric: true,
      render: (row) => <RatingValue figure={row.pace} n={n} label="pace" />,
    },
    {
      key: "form",
      label: "Last 10",
      value: (row) => row.form.join(""),
      render: (row) => <FormDots form={row.form} />,
    },
    {
      key: "streak",
      label: "Streak",
      value: (row) => row.streak,
      numeric: true,
      render: (row) => (row.streak === 0 ? "—" : `${row.streak > 0 ? "W" : "L"}${Math.abs(row.streak)}`),
    },
  ];

  const topPlayers = (team: string) =>
    players
      .filter((player) => player.team === team)
      .sort((a, b) => b.points_per_game - a.points_per_game)
      .slice(0, 3);

  return (
    <div data-testid="teams-table">
    <StatTable<Row>
      rows={rows}
      columns={columns}
      rowKey={(row) => row.team}
      initialSort={
        initialSortKey === "power"
          ? { key: "power", dir: "desc" }
          : { key: "seed", dir: "asc" }
      }
      caption="NBA standings: seed, record, ratings and form"
      expand={(row) => {
        const nextGame = overview.next_game?.[row.team] ?? null;
        return (
        <div className="grid gap-4 sm:grid-cols-2" data-testid={`team-detail-${row.team}`}>
          <div>
            <h3 className="mb-1 text-xs uppercase tracking-wide text-pr-text-dim">Ratings</h3>
            {/* The same eleven metrics the game detail shows, from the same
                payload and the same pieces -- not a second list of four. */}
            <RatingList
              metrics={overview.metrics}
              figure={(metric) => overview.ratings?.[row.team]?.[metric.key]}
              n={n}
              team={row.team}
              testId={`team-ratings-${row.team}`}
            />
          </div>
          <div>
            <h3 className="mb-1 text-xs uppercase tracking-wide text-pr-text-dim">Last five</h3>
            <ul className="text-sm">
              {(overview.last_games?.[row.team] ?? []).map((game) => (
                <li key={game.game_id} className="flex items-center justify-between gap-2">
                  <span className={game.result === "W" ? "text-pr-win" : "text-pr-loss"}>
                    {game.result} {game.opponent}
                  </span>
                  <span className="tabular-nums text-pr-text-dim">{game.score}</span>
                </li>
              ))}
            </ul>
            <h3 className="mb-1 mt-3 text-xs uppercase tracking-wide text-pr-text-dim">Next game</h3>
            {nextGame ? (
              <p className="text-sm">
                vs {nextGame.opponent}
                {nextGame.pick
                  ? ` · pick: ${nextGame.pick} ${((nextGame.pick_probability ?? 0) * 100).toFixed(0)}%`
                  : " · no pick yet"}
              </p>
            ) : (
              <p className="text-sm text-pr-text-faint">No game scheduled.</p>
            )}
            {topPlayers(row.team).length > 0 && (
              <h3 className="mb-1 mt-3 text-xs uppercase tracking-wide text-pr-text-dim">Top scorers</h3>
            )}
            <ul className="text-sm">
              {topPlayers(row.team).map((player) => (
                <li key={player.player_id}>
                  <Link to={`/players?team=${row.team}`} className="underline underline-offset-2">
                    {player.player_name}
                  </Link>{" "}
                  <span className="tabular-nums text-pr-text-dim">
                    {player.points_per_game.toFixed(1)} ppg
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </div>
        );
      }}
    />
    </div>
  );
}
