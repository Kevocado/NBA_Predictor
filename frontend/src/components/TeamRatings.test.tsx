import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import TeamRatings from "./TeamRatings";
import type { GameRatings } from "../api/client";

/** The metrics the backend serves, in the order a reader reads them. */
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

const figure = (value: number | null, rank: number | null) => ({ value, rank });

const ratings: GameRatings = {
  as_of: "2026-10-21",
  n_teams: 30,
  caption: "Last 15 games each. Early season includes last season's games.",
  metrics: METRICS,
  teams: {
    BOS: Object.fromEntries(METRICS.map((m) => [m.key, figure(100, 5)])),
    DET: Object.fromEntries(METRICS.map((m) => [m.key, figure(101, 4)])),
  },
};

describe("TeamRatings", () => {
  it("renders one row per metric, with both teams' figures", () => {
    render(<TeamRatings ratings={ratings} awayTeam="BOS" homeTeam="DET" />);

    const rows = screen.getAllByTestId(/^rating-row-/);
    expect(rows).toHaveLength(METRICS.length);
    expect(screen.getByTestId("rating-row-net")).toBeInTheDocument();
    expect(screen.getByTestId("rating-row-reb")).toBeInTheDocument();
  });

  it("shows the figure and the place in the league, not the place alone", () => {
    render(<TeamRatings ratings={ratings} awayTeam="BOS" homeTeam="DET" />);

    const row = screen.getByTestId("rating-row-net");
    const values = within(row).getAllByTestId("rating-value").map((n) => n.textContent);
    const ranks = within(row).getAllByTestId("rating-rank").map((n) => n.textContent);
    expect(values).toEqual(["100", "101"]);
    expect(ranks).toEqual(["5", "4"]);
  });

  it("colours the rank by the kit's scale, so there is one colour scale everywhere", () => {
    render(<TeamRatings ratings={ratings} awayTeam="BOS" homeTeam="DET" />);

    // 4th of 30 is the top third; 5th too. The chip says which.
    const chips = within(screen.getByTestId("rating-row-net")).getAllByTestId("rating-rank");
    for (const chip of chips) {
      expect(chip.className).toMatch(/pr-rank-(good|mid|bad)/);
    }
    expect(chips[0].className).toMatch(/pr-rank-good/);
  });

  it("says what the window was, so a rating can be checked", () => {
    render(<TeamRatings ratings={ratings} awayTeam="BOS" homeTeam="DET" />);

    expect(screen.getByText(/Last 15 games each/)).toBeInTheDocument();
    expect(screen.getByText(/last season's games/)).toBeInTheDocument();
  });

  it("shows an em dash for a figure nobody has, never a zero", () => {
    const withAGap: GameRatings = {
      ...ratings,
      teams: {
        ...ratings.teams,
        BOS: { ...ratings.teams.BOS, fg3_pct: figure(null, null) },
      },
    };
    render(<TeamRatings ratings={withAGap} awayTeam="BOS" homeTeam="DET" />);

    const row = screen.getByTestId("rating-row-fg3_pct");
    expect(within(row).getAllByTestId("rating-missing")).toHaveLength(1);
    expect(within(row).getAllByTestId("rating-value")).toHaveLength(1);
  });

  it("puts the away team first and the home team last", () => {
    render(<TeamRatings ratings={ratings} awayTeam="BOS" homeTeam="DET" />);

    const headers = screen.getAllByRole("columnheader").map((h) => h.textContent);
    expect(headers[0]).toContain("BOS");
    expect(headers[1]).toContain("Stat");
    expect(headers[2]).toContain("DET");
  });
});
