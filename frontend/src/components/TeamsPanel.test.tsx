import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import TeamsPanel from "./TeamsPanel";
import { api, type PlayerHubRow, type PowerRankingRow, type StandingsRow, type TeamHubRow, type TeamOverview } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    getHubStandings: vi.fn(),
    getHubRankings: vi.fn(),
    getHubTeams: vi.fn(),
    getHubTeamOverview: vi.fn(),
    getHubPlayers: vi.fn(),
  },
}));

const standings: StandingsRow[] = [
  { conference: "East", seed: 1, abbreviation: "BOS", wins: 4, losses: 0, win_pct: 1, games_back: 0, playoff_status: "clinched", season_state: "regular" },
  { conference: "East", seed: 2, abbreviation: "MIA", wins: 3, losses: 1, win_pct: 0.75, games_back: 1, playoff_status: "clinched", season_state: "regular" },
  { conference: "West", seed: 1, abbreviation: "LAL", wins: 2, losses: 2, win_pct: 0.5, games_back: 0, playoff_status: "clinched", season_state: "regular" },
];

const rankings: PowerRankingRow[] = [
  { rank: 1, abbreviation: "LAL", power_rating: 1512.4, trend: "up" },
  { rank: 2, abbreviation: "BOS", power_rating: 1508.1, trend: "down" },
  { rank: 3, abbreviation: "MIA", power_rating: 1490.2, trend: "steady" },
];

const teams: TeamHubRow[] = [
  { abbreviation: "BOS", conference: "East", division: "Atlantic", wins: 4, losses: 0, points_per_game: 112, opp_points_per_game: 106, net_rating: 6, pace: 99, streak: 3 },
  { abbreviation: "MIA", conference: "East", division: "Atlantic", wins: 3, losses: 1, points_per_game: 108, opp_points_per_game: 108, net_rating: 0, pace: 101, streak: 1 },
  { abbreviation: "LAL", conference: "West", division: "Pacific", wins: 2, losses: 2, points_per_game: 115, opp_points_per_game: 110, net_rating: 5, pace: 97, streak: 2 },
];

const figure = (value: number | null, rank: number | null) => ({ value, rank });

const METRICS = [
  { key: "net", label: "Net rating", unit: "", help: "Points scored minus points allowed, per 100 possessions.", higher_is_better: true },
  { key: "ortg", label: "Offence", unit: "pts per 100", help: "Points scored per 100 possessions.", higher_is_better: true },
  { key: "drtg", label: "Defence", unit: "allowed per 100", help: "Points allowed per 100 possessions.", higher_is_better: false },
  { key: "pace", label: "Pace", unit: "possessions", help: "Possessions per game.", higher_is_better: true },
  { key: "efg_pct", label: "eFG%", unit: "", help: "Points per shot.", higher_is_better: true },
  { key: "tov_rate", label: "Turnover %", unit: "", help: "Turnovers per 100 possessions.", higher_is_better: false },
  { key: "orb_pct", label: "Offensive rebounds %", unit: "", help: "Share of missed shots got back.", higher_is_better: true },
  { key: "ft_rate", label: "Free-throw rate", unit: "", help: "Free throws per 100 possessions.", higher_is_better: true },
  { key: "fg3_pct", label: "3P%", unit: "", help: "Share of three-point shots made.", higher_is_better: true },
  { key: "fg3_pct_allowed", label: "3P% allowed", unit: "", help: "Share of three-point shots the opponents made.", higher_is_better: false },
  { key: "reb", label: "Rebounds", unit: "", help: "Rebounds per game.", higher_is_better: true },
];

/** Each metric's figure and rank per team: the league the fixture describes. */
const VALUES: Record<string, Record<string, number>> = {
  net: { BOS: 6.1, MIA: 0.2, LAL: 4.8 },
  ortg: { BOS: 112.4, MIA: 108.0, LAL: 115.0 },
  drtg: { BOS: 106.3, MIA: 107.8, LAL: 110.2 },
  pace: { BOS: 99.1, MIA: 101.2, LAL: 97.0 },
  efg_pct: { BOS: 55.1, MIA: 53.4, LAL: 56.0 },
  tov_rate: { BOS: 12.9, MIA: 11.4, LAL: 13.6 },
  orb_pct: { BOS: 25.4, MIA: 26.8, LAL: 23.1 },
  ft_rate: { BOS: 21.2, MIA: 24.7, LAL: 19.8 },
  fg3_pct: { BOS: 37.2, MIA: 38.4, LAL: 39.1 },
  fg3_pct_allowed: { BOS: 35.8, MIA: 36.9, LAL: 34.4 },
  reb: { BOS: 44.9, MIA: 42.1, LAL: 46.3 },
};

const RANKS: Record<string, Record<string, number>> = {
  net: { BOS: 4, MIA: 15, LAL: 6 },
  ortg: { BOS: 2, MIA: 11, LAL: 1 },
  drtg: { BOS: 1, MIA: 9, LAL: 18 },
  pace: { BOS: 14, MIA: 1, LAL: 22 },
  efg_pct: { BOS: 3, MIA: 8, LAL: 5 },
  tov_rate: { BOS: 6, MIA: 2, LAL: 9 },
  orb_pct: { BOS: 7, MIA: 5, LAL: 12 },
  ft_rate: { BOS: 11, MIA: 4, LAL: 19 },
  fg3_pct: { BOS: 9, MIA: 6, LAL: 2 },
  fg3_pct_allowed: { BOS: 5, MIA: 10, LAL: 21 },
  reb: { BOS: 3, MIA: 12, LAL: 8 },
};

const ratingFigure = (value: number, rank: number) => ({ value, rank });

const overview: TeamOverview = {
  as_of: "2026-11-01",
  n_teams: 30,
  caption: "Last 15 games each.",
  metrics: METRICS,
  ratings: Object.fromEntries(
    ["BOS", "MIA", "LAL"].map((team) => [
      team,
      Object.fromEntries(
        METRICS.map((metric) => [metric.key, ratingFigure(VALUES[metric.key][team], RANKS[metric.key][team])]),
      ),
    ]),
  ),
  form: { BOS: ["W", "W", "L"], MIA: ["W", "L"], LAL: ["L", "W", "W"] },
  last_games: {
    BOS: [{ game_id: "g1", game_date: "2026-10-30", opponent: "MIA", home: true, score: "110–100", result: "W" }],
    MIA: [],
    LAL: [],
  },
  next_game: {
    BOS: { game_id: "g2", game_date: "2026-11-02", tip_off: null, opponent: "MIA", home: true, pick: "BOS", pick_probability: 0.58 },
    MIA: null,
    LAL: null,
  },
};

const players: PlayerHubRow[] = [
  { player_id: "p1", player_name: "Jayson Tatum", team: "BOS", position: "F", rating: 20, live_form_rating: 21, points_per_game: 28.4, rebounds_per_game: 8.1, assists_per_game: 4.2, fg_pct: 0.48, three_pt_pct: 0.39, ft_pct: 0.85, usage_rate: 0.29, minutes_per_game: 34 },
  { player_id: "p2", player_name: "Jaylen Brown", team: "BOS", position: "G", rating: 18, live_form_rating: 19, points_per_game: 26.1, rebounds_per_game: 4.4, assists_per_game: 3.7, fg_pct: 0.47, three_pt_pct: 0.36, ft_pct: 0.79, usage_rate: 0.28, minutes_per_game: 33 },
  { player_id: "p3", player_name: "Derrick White", team: "BOS", position: "G", rating: 14, live_form_rating: 15, points_per_game: 16.2, rebounds_per_game: 3.9, assists_per_game: 5.1, fg_pct: 0.52, three_pt_pct: 0.41, ft_pct: 0.8, usage_rate: 0.18, minutes_per_game: 30 },
];

function renderPanel(path = "/hub") {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <TeamsPanel />
    </MemoryRouter>,
  );
}

describe("TeamsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getHubStandings).mockResolvedValue(standings);
    vi.mocked(api.getHubRankings).mockResolvedValue(rankings);
    vi.mocked(api.getHubTeams).mockResolvedValue(teams);
    vi.mocked(api.getHubTeamOverview).mockResolvedValue(overview);
    vi.mocked(api.getHubPlayers).mockResolvedValue(players);
  });

  it("shows one table with every team on it", async () => {
    renderPanel();

    await screen.findByTestId("teams-panel");
    const rows = within(screen.getByRole("table")).getAllByRole("row");
    // The header row plus one row per team.
    expect(rows).toHaveLength(4);
  });

  it("sorts by seed, and by power rating, and by net rating", async () => {
    renderPanel();
    await screen.findByTestId("teams-panel");
    const bodyRows = () =>
      within(screen.getByRole("table"))
        .getAllByRole("row")
        .slice(1)
        .map((row) => within(row).getAllByRole("cell")[0].textContent);

    // The default is the standings order.
    expect(bodyRows()[0]).toContain("Celtics");

    await userEvent.click(screen.getByRole("button", { name: /power/i }));
    const byPower = bodyRows();
    expect(byPower[0]).toContain("Lakers");

    await userEvent.click(screen.getByRole("button", { name: /^net$/i }));
    const byNet = bodyRows();
    expect(byNet[0]).toContain("Celtics");
  });

  it("shows one conference at a time, and the league by default", async () => {
    renderPanel();
    await screen.findByTestId("teams-panel");
    const group = screen.getByRole("group", { name: "Conference" });

    const table = () => screen.getByRole("table");
    await userEvent.click(within(group).getByRole("button", { name: "West" }));
    expect(within(table()).queryByText("Celtics")).toBeNull();
    expect(within(table()).getByText("Lakers")).toBeInTheDocument();

    await userEvent.click(within(group).getByRole("button", { name: "East" }));
    expect(within(table()).getByText("Celtics")).toBeInTheDocument();
    expect(within(table()).queryByText("Lakers")).toBeNull();

    await userEvent.click(within(group).getByRole("button", { name: "League" }));
    expect(within(table()).getByText("Celtics")).toBeInTheDocument();
    expect(within(table()).getByText("Lakers")).toBeInTheDocument();
  });

  it("names the league's best at each thing", async () => {
    renderPanel();
    await screen.findByTestId("teams-panel");
    const tiles = screen.getByTestId("leader-tiles");
    expect(within(tiles).getByText("Best offence")).toBeInTheDocument();
    expect(within(tiles).getByText("Best defence")).toBeInTheDocument();
    expect(within(tiles).getByText("Fastest pace")).toBeInTheDocument();
    expect(within(tiles).getByText("Highest power rating")).toBeInTheDocument();
    // The best offence is the team ranked first on it.
    const tile = (title: string) => within(tiles).getByText(title).parentElement as HTMLElement;
    expect(within(tile("Best offence")).getByText("Lakers")).toBeInTheDocument();
    expect(within(tile("Best offence")).getByText("115.0")).toBeInTheDocument();
    // The fastest pace is the team ranked first on pace.
    expect(within(tile("Fastest pace")).getByText("101.2")).toBeInTheDocument();
    // The best defence is the team ranked first on it -- a defence is ranked on
    // what it ALLOWS, so the first place is the team that concedes least.
    expect(within(tile("Best defence")).getByText("Celtics")).toBeInTheDocument();
  });

  it("opens a team's detail: its ratings, its last five, its next game and its top scorers", async () => {
    renderPanel();
    await screen.findByTestId("teams-panel");

    await userEvent.click(screen.getByRole("button", { name: /Celtics/i }));

    const detail = screen.getByTestId("team-detail-BOS");
    expect(within(detail).getByText("Net rating")).toBeInTheDocument();
    expect(within(detail).getByText(/pick: BOS 58%/)).toBeInTheDocument();
    expect(within(detail).getByText("Jayson Tatum")).toBeInTheDocument();
    expect(within(detail).getByText(/28\.4 ppg/)).toBeInTheDocument();
  });

  it("shows all eleven ratings in a team's row, each with its rank", async () => {
    renderPanel();
    await screen.findByTestId("teams-panel");

    await userEvent.click(screen.getByRole("button", { name: /Celtics/i }));

    const ratings = screen.getByTestId("team-ratings-BOS");
    const rows = within(ratings).getAllByTestId(/^rating-row-/);
    expect(rows).toHaveLength(11);
    for (const row of rows) {
      expect(within(row).getByTestId("rating-rank")).toBeInTheDocument();
    }
    // The same eleven metrics the game detail shows.
    expect(within(ratings).getByText("eFG%")).toBeInTheDocument();
    expect(within(ratings).getByText("3P% allowed")).toBeInTheDocument();
  });

  it("does not call a four-game lead 'clinched' before the league has settled", async () => {
    vi.mocked(api.getHubStandings).mockResolvedValue([
      { abbreviation: "BOS", conference: "East", seed: 1, wins: 4, losses: 0, win_pct: 1, games_back: 0, playoff_status: "clinched", season_state: "regular" },
      { abbreviation: "MIA", conference: "East", seed: 2, wins: 1, losses: 3, win_pct: 0.25, games_back: 3, playoff_status: "eliminated", season_state: "regular" },
    ] as never);

    renderPanel();
    await screen.findByTestId("teams-panel");

    expect(screen.getByText("Leading")).toBeInTheDocument();
    expect(screen.getByText("Trailing")).toBeInTheDocument();
    expect(screen.queryByText("Clinched")).toBeNull();
  });

  it("sorts by power rating when the old power-rankings link asks for it", async () => {
    renderPanel("/hub?sort=power");
    await screen.findByTestId("teams-panel");
    const firstRow = within(screen.getByRole("table")).getAllByRole("row")[1];
    expect(within(firstRow).getAllByRole("cell")[0].textContent).toContain("Lakers");
  });
});
