import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { render, screen, within } from "@testing-library/react";
import PlayerBoxScore from "./PlayerBoxScore";
import {
  STAT_KEYS,
  attachHubRows,
  hasActuals,
  orderPlayers,
  pivotProps,
  splitByTeam,
  sumStat,
} from "../lib/boxScore";
import type { PlayerHubRow, PlayerProp } from "../api/client";

const STATS = ["points", "rebounds", "assists", "threes"] as const;

/** vitest runs with cwd at frontend/, and import.meta.url is not a file URL here. */
function readCss(): string {
  return readFileSync(resolve(process.cwd(), "src/index.css"), "utf8");
}

function propsFor(playerId: string, name: string, values: number[], actual: number[] | null = null): PlayerProp[] {
  return STATS.map((stat, i) => ({
    player_id: playerId,
    player_name: name,
    stat,
    predicted_value: values[i],
    actual_value: actual ? actual[i] : null,
    rebuilt: false,
  }));
}

function hubRow(playerId: string, name: string, team: string, position: string): PlayerHubRow {
  return {
    player_id: playerId, player_name: name, team, position,
    rating: 0, live_form_rating: 0, points_per_game: 0, rebounds_per_game: 0,
    assists_per_game: 0, fg_pct: 0, three_pt_pct: 0, ft_pct: 0, usage_rate: 0, minutes_per_game: 0,
  };
}

/** Two teams of three, the shape the live feed actually has. */
const AWAY = [
  { id: "a1", name: "A One", vals: [20, 5, 6, 2] },
  { id: "a2", name: "A Two", vals: [10, 8, 3, 1] },
  { id: "a3", name: "A Three", vals: [5, 2, 1, 0] },
] as const;
const HOME = [
  { id: "h1", name: "H One", vals: [30, 10, 12, 4] },
  { id: "h2", name: "H Two", vals: [15, 3, 2, 1] },
  { id: "h3", name: "H Three", vals: [8, 1, 1, 0] },
] as const;

const allProps: PlayerProp[] = [
  ...AWAY.flatMap((p) => propsFor(p.id, p.name, [...p.vals])),
  ...HOME.flatMap((p) => propsFor(p.id, p.name, [...p.vals])),
];
const allHub: PlayerHubRow[] = [
  ...AWAY.map((p, i) => hubRow(p.id, p.name, "MIA", i === 2 ? "C" : "G")),
  ...HOME.map((p, i) => hubRow(p.id, p.name, "BOS", i === 2 ? "C" : "F")),
];

describe("boxScore pivot", () => {
  it("collapses one row per player per stat into one row per player", () => {
    expect(allProps).toHaveLength(24);
    const rows = pivotProps(allProps);
    expect(rows).toHaveLength(6);
    expect(rows.find((r) => r.player_id === "a1")?.predicted).toEqual({
      points: 20, rebounds: 5, assists: 6, threes: 2,
    });
  });

  it("leaves an unplayed stat absent rather than defaulting it to 0", () => {
    const rows = pivotProps(propsFor("x", "X", [10, 0, 0, 0]));
    expect(rows[0].actual).toEqual({});
    expect(hasActuals(rows)).toBe(false);
  });

  it("ignores a stat key it does not have a column for", () => {
    const rows = pivotProps([
      ...propsFor("x", "X", [10, 5, 5, 1]),
      { player_id: "x", player_name: "X", stat: "blocks", predicted_value: 3, actual_value: null, rebuilt: false },
    ]);
    // The four known stats survive; "blocks" is dropped rather than shown as an
    // unlabelled column.
    expect(rows[0].predicted).toEqual({ points: 10, rebounds: 5, assists: 5, threes: 1 });
  });

  it("produces no row at all for a player who only has unknown stats", () => {
    const rows = pivotProps([
      { player_id: "x", player_name: "X", stat: "blocks", predicted_value: 3, actual_value: null, rebuilt: false },
    ]);
    expect(rows).toEqual([]);
  });

  it("keeps only the first prop for a repeated player and stat", () => {
    const rows = pivotProps([
      ...propsFor("x", "X", [10, 1, 1, 1]),
      { player_id: "x", player_name: "X", stat: "points", predicted_value: 99, actual_value: null, rebuilt: false },
    ]);
    expect(rows[0].predicted.points).toBe(10);
  });
});

describe("boxScore team split", () => {
  it("puts each player on the side their team says, not by array order", () => {
    const rows = attachHubRows(pivotProps(allProps), allHub);
    const { away, home, unattributed } = splitByTeam(rows, "BOS", "MIA");
    expect(away.map((r) => r.player_id)).toEqual(["a1", "a2", "a3"]);
    expect(home.map((r) => r.player_id)).toEqual(["h1", "h2", "h3"]);
    expect(unattributed).toHaveLength(0);
  });

  it("returns a player it cannot place instead of guessing a side", () => {
    const hub = [...allHub.filter((p) => p.player_id !== "a2")];
    const rows = attachHubRows(pivotProps(allProps), hub);
    const { away, home, unattributed } = splitByTeam(rows, "BOS", "MIA");
    expect(away.map((r) => r.player_id)).not.toContain("a2");
    expect(home.map((r) => r.player_id)).not.toContain("a2");
    expect(unattributed.map((r) => r.player_id)).toEqual(["a2"]);
  });

  it("orders guards, then wings, then centres, and the biggest role first within a position", () => {
    const rows = attachHubRows(pivotProps(allProps), allHub);
    const { home } = splitByTeam(rows, "BOS", "MIA");
    // h1/h2 are F, h3 is C -- so the forwards lead and the centre is last.
    expect(home.map((r) => r.position)).toEqual(["F", "F", "C"]);
    expect(home[0].player_id).toBe("h1"); // 30 points beats 15
  });

  it("sorts an unrecognised position last rather than first", () => {
    const hub = allHub.map((p) => (p.player_id === "h3" ? { ...p, position: "" } : p));
    const rows = attachHubRows(pivotProps(allProps), hub);
    expect(splitByTeam(rows, "BOS", "MIA").home.map((r) => r.player_id)).toEqual(["h1", "h2", "h3"]);
  });

  it("does not mutate its input", () => {
    const rows = attachHubRows(pivotProps(allProps), allHub);
    const before = rows.map((r) => r.player_id);
    orderPlayers(rows);
    expect(rows.map((r) => r.player_id)).toEqual(before);
  });
});

describe("boxScore totals", () => {
  it("sums each side's own column", () => {
    const rows = attachHubRows(pivotProps(allProps), allHub);
    const { away, home } = splitByTeam(rows, "BOS", "MIA");
    expect(sumStat(away, "points", "predicted")).toBe(35);
    expect(sumStat(home, "points", "predicted")).toBe(53);
    expect(sumStat(away, "rebounds", "predicted")).toBe(15);
  });
});

describe("PlayerBoxScore", () => {
  function renderBox(overrides: { props?: PlayerProp[]; hub?: PlayerHubRow[] } = {}) {
    return render(
      <PlayerBoxScore
        playerProps={overrides.props ?? allProps}
        hubPlayers={overrides.hub ?? allHub}
        homeTeam="BOS"
        awayTeam="MIA"
      />,
    );
  }

  it("renders one row per player per side, not one row per prop", () => {
    renderBox();
    // 6 players, 3 per side, so 3 body rows -- down from 24 prop rows.
    expect(screen.getAllByTestId("box-score-row")).toHaveLength(3);
  });

  it("names both teams as real column headers", () => {
    renderBox();
    // teamName gives the short nickname: "Heat", "Celtics".
    expect(screen.getByRole("columnheader", { name: "Heat" })).toBeTruthy();
    expect(screen.getByRole("columnheader", { name: "Celtics" })).toBeTruthy();
  });

  it("uses real table semantics with scopes", () => {
    renderBox();
    const table = screen.getByRole("table");
    expect(table.tagName).toBe("TABLE");
    // col headers for the stat columns, on both sides.
    for (const label of ["Pts", "Reb", "Ast", "3PM"]) {
      expect(screen.getAllByRole("columnheader", { name: label })).toHaveLength(2);
    }
    // row headers are the player names: two per row, one per side.
    expect(screen.getByRole("rowheader", { name: "A One" })).toBeTruthy();
    expect(screen.getByRole("rowheader", { name: "H One" })).toBeTruthy();
  });

  it("shows the predicted value, which is the whole point of the page", () => {
    renderBox();
    const row = screen.getAllByTestId("box-score-row")[0];
    expect(within(row).getByRole("rowheader", { name: "A One" })).toBeTruthy();
    expect(within(row).getByText("20.0")).toBeTruthy();
  });

  it("keeps both sides of a row in the same table row, so they cannot drift apart", () => {
    renderBox();
    const rows = screen.getAllByTestId("box-score-row");
    rows.forEach((row) => {
      // Two row headers per body row: the away name and the home name.
      expect(within(row).getAllByRole("rowheader")).toHaveLength(2);
    });
  });

  it("pads a short side so the tall side keeps its place and totals sit under both", () => {
    // Drop one player from the home side only, so the home side is one short
    // and the last row is padding on one side and a real player on the other.
    const hub = allHub.filter((p) => p.player_id !== "h3");
    renderBox({ hub });
    const rows = screen.getAllByTestId("box-score-row");
    expect(rows).toHaveLength(3);
    const last = within(rows[2]).getAllByRole("rowheader");
    expect(last.map((cell) => cell.textContent)).toEqual(["A Three", ""]);
  });

  it("totals each side from its own players only", () => {
    renderBox();
    const totals = screen.getByTestId("box-score-totals");
    // MIA 20+10+5 = 35 points, BOS 30+15+8 = 53. Neither side gets the
    // other's players, which is the failure a shared total would hide.
    expect(within(totals).getByText("35.0")).toBeTruthy();
    expect(within(totals).getByText("53.0")).toBeTruthy();
  });

  it("says so when a player cannot be matched to either team", () => {
    const hub = allHub.filter((p) => p.player_id !== "a2");
    renderBox({ hub });
    expect(screen.getByText(/could not be matched/i)).toBeTruthy();
  });

  it("shows the actual under the prediction once the game is played", () => {
    const played: PlayerProp[] = [
      ...AWAY.flatMap((p) => propsFor(p.id, p.name, [...p.vals], [22, 4, 5, 3])),
      ...HOME.flatMap((p) => propsFor(p.id, p.name, [...p.vals], [28, 9, 11, 5])),
    ];
    renderBox({ props: played });
    const row = screen.getAllByTestId("box-score-row")[0];
    expect(within(row).getByText("20.0")).toBeTruthy(); // predicted
    // Actual with the miss against the prediction -- the accuracy signal the
    // old flat list carried as "off by 2.0".
    expect(within(row).getByText("22 (+2.0)")).toBeTruthy();
    // and an actual totals row appears, because a team total is the number a
    // reader wants and it would be wrong to leave them adding it up.
    expect(screen.getByTestId("box-score-actual-totals")).toBeTruthy();
  });

  it("scores a rebuilt projection never: no delta, even when it is wrong", () => {
    const rebuilt: PlayerProp[] = [
      ...AWAY.flatMap((p) =>
        propsFor(p.id, p.name, [...p.vals], [22, 4, 5, 3]).map((prop) => ({ ...prop, rebuilt: true })),
      ),
      ...HOME.flatMap((p) => propsFor(p.id, p.name, [...p.vals], [28, 9, 11, 5])),
    ];
    renderBox({ props: rebuilt });
    const row = screen.getAllByTestId("box-score-row")[0];
    // Away cells are predicted-then-actual, no delta. A One predicted
    // 20/5/6/2 and the actuals were 22/4/5/3.
    const awayCells = within(row).getAllByRole("cell").slice(0, 4);
    expect(awayCells.map((c) => c.textContent)).toEqual(["20.022", "5.04", "6.05", "2.03"]);
    // The home side of this same row is NOT rebuilt, so its delta is still
    // there. That is the control: it proves the absence above is the rebuilt
    // rule and not a missing feature.
    const homeCells = within(row).getAllByRole("cell").slice(4);
    expect(homeCells[0].textContent?.replace(/\s+/g, "")).toBe("30.028(-2.0)");
  });

  it("scrolls on both axes with a bounded height, so the card does not grow", () => {
    renderBox();
    // jsdom loads no stylesheet, so getComputedStyle returns "" for everything
    // and getBoundingClientRect returns zeros -- neither can see the layout.
    // Asserting on either would be theatre. So: assert the element carries the
    // hook, and assert the stylesheet actually declares the contract. Real
    // geometry is measured in a browser.
    const scroller = screen.getByTestId("box-score-scroll");
    expect(scroller.className).toBe("box-score-scroll");
    expect(scroller.className).not.toContain("overflow-auto"); // no utility, no drift

    const css = readCss();
    const rule = css.slice(css.indexOf(".box-score-scroll {"));
    expect(rule.slice(0, rule.indexOf("}"))).toMatch(/max-height:\s*[\d.]+rem/);
    expect(rule.slice(0, rule.indexOf("}"))).toMatch(/overflow:\s*auto/);
  });

  it("sticks the headers and totals so a scrolled split table stays readable", () => {
    const css = readCss();
    // Without this, the column labels scroll out of view and a two-sided
    // comparison becomes guesswork.
    expect(css).toMatch(/\.box-score-table thead th[\s\S]*?position:\s*sticky/);
    expect(css).toMatch(/\.box-score-table tfoot th[\s\S]*?position:\s*sticky/);
    // Fixed layout plus a symmetric colgroup is what keeps the two halves the
    // same width; with auto layout the halves drift apart.
    expect(css).toMatch(/\.box-score-table \{[\s\S]*?table-layout:\s*fixed/);
  });

  it("does not grow its row count with the player count", () => {
    const big: PlayerProp[] = [];
    const bigHub: PlayerHubRow[] = [];
    for (let i = 0; i < 11; i++) {
      big.push(...propsFor(`m${i}`, `Mia ${i}`, [10 + i, 2, 2, 1]));
      bigHub.push(hubRow(`m${i}`, `Mia ${i}`, "MIA", "G"));
      big.push(...propsFor(`b${i}`, `Bos ${i}`, [10 + i, 2, 2, 1]));
      bigHub.push(hubRow(`b${i}`, `Bos ${i}`, "BOS", "G"));
    }
    renderBox({ props: big, hub: bigHub });
    // 22 players, 11 per side: 11 rows. Bounded by the container's max-height,
    // not by the number of players.
    expect(screen.getAllByTestId("box-score-row")).toHaveLength(11);
  });

  it("renders nothing when there are no props", () => {
    const { container } = renderBox({ props: [] });
    expect(container.querySelector("[data-testid='player-box-score']")).toBeNull();
  });

  it("covers all four stats the feed actually carries", () => {
    expect(STAT_KEYS).toEqual(["points", "rebounds", "assists", "threes"]);
  });
});
