import { describe, expect, it } from "vitest";
import {
  DEFAULT_FILTERS,
  MIN_MINUTES,
  filterPlayers,
  leadersOn,
  matchesPosition,
  type PlayerFilters,
} from "./playerFilters";
import type { PlayerHubRow } from "../api/client";

const player = (over: Partial<PlayerHubRow> & { player_id: string }): PlayerHubRow => ({
  player_name: `Player ${over.player_id}`,
  team: "BOS",
  position: "G",
  rating: 10,
  live_form_rating: 10,
  points_per_game: 10,
  rebounds_per_game: 5,
  assists_per_game: 3,
  fg_pct: 0.47,
  three_pt_pct: 0.37,
  ft_pct: 0.85,
  usage_rate: 0.24,
  minutes_per_game: 30,
  ...over,
});

const filters = (over: Partial<PlayerFilters> = {}): PlayerFilters => ({ ...DEFAULT_FILTERS, ...over });

describe("playerFilters", () => {
  it("hides the players who do not play, unless the switch is off", () => {
    const rows = [player({ player_id: "1" }), player({ player_id: "2", minutes_per_game: 4 })];

    expect(filterPlayers(rows, filters()).map((r) => r.player_id)).toEqual(["1"]);
    expect(
      filterPlayers(rows, filters({ minMinutes: false })).map((r) => r.player_id),
    ).toEqual(["1", "2"]);
  });

  it("keeps a player who plays exactly the minimum", () => {
    const rows = [player({ player_id: "1", minutes_per_game: MIN_MINUTES })];

    expect(filterPlayers(rows, filters())).toHaveLength(1);
  });

  it("searches by name and by team", () => {
    const rows = [
      player({ player_id: "1", player_name: "Jayson Tatum", team: "BOS" }),
      player({ player_id: "2", player_name: "Jaylen Brown", team: "BOS" }),
      player({ player_id: "3", player_name: "LeBron James", team: "LAL" }),
    ];

    expect(filterPlayers(rows, filters({ search: "tatum" })).map((r) => r.player_id)).toEqual(["1"]);
    expect(filterPlayers(rows, filters({ search: "lal" })).map((r) => r.player_id)).toEqual(["3"]);
  });

  it("filters by team", () => {
    const rows = [player({ player_id: "1", team: "BOS" }), player({ player_id: "2", team: "LAL" })];

    expect(filterPlayers(rows, filters({ team: "LAL" })).map((r) => r.player_id)).toEqual(["2"]);
  });

  it("puts a guard in G, a forward in F and a centre in C", () => {
    expect(matchesPosition("G", "G")).toBe(true);
    expect(matchesPosition("PG", "G")).toBe(true);
    expect(matchesPosition("SF", "F")).toBe(true);
    expect(matchesPosition("C", "C")).toBe(true);
    expect(matchesPosition("G", "C")).toBe(false);
    expect(matchesPosition("", "C")).toBe(false);
    expect(matchesPosition("", "All")).toBe(true);
  });

  it("filters by position", () => {
    const rows = [
      player({ player_id: "1", position: "PG" }),
      player({ player_id: "2", position: "SF" }),
      player({ player_id: "3", position: "C" }),
    ];

    expect(filterPlayers(rows, filters({ position: "G" })).map((r) => r.player_id)).toEqual(["1"]);
    expect(filterPlayers(rows, filters({ position: "F" })).map((r) => r.player_id)).toEqual(["2"]);
    expect(filterPlayers(rows, filters({ position: "C" })).map((r) => r.player_id)).toEqual(["3"]);
    expect(filterPlayers(rows, filters({ position: "All" })).map((r) => r.player_id)).toEqual(["1", "2", "3"]);
  });

  it("names the five biggest on a stat, and skips the players with no figure", () => {
    const rows = Array.from({ length: 7 }, (_, i) =>
      player({ player_id: `${i}`, points_per_game: i }),
    );

    const leaders = leadersOn(rows, (row) => row.points_per_game);

    expect(leaders.map((r) => r.player_id)).toEqual(["6", "5", "4", "3", "2"]);
    expect(leadersOn(rows, (row) => row.threes_per_game ?? null)).toEqual([]);
  });
});
