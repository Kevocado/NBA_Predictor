import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { TopCalls, buildTopCalls } from "./TopCalls";
import { api } from "../api/client";
import type { OutPlayer as ApiOutPlayer, PlayerProp } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    getGamePlayers: vi.fn(),
    getGameOutPlayers: vi.fn(),
  },
}));

/** One prop row. `mae` is omitted rather than defaulted so a test that means
 *  "the API sent no estimate" cannot accidentally send 0. */
function prop(
  playerId: string,
  name: string,
  stat: string,
  predicted: number,
  mae?: number | null,
  actual: number | null = null,
): PlayerProp {
  const row: PlayerProp = {
    player_id: playerId,
    player_name: name,
    stat,
    predicted_value: predicted,
    actual_value: actual,
  };
  if (mae !== undefined) row.mae = mae;
  return row;
}

function outRow(playerId: string, name: string, team = "BOS"): ApiOutPlayer {
  return {
    player_id: playerId,
    player_name: name,
    team,
    status: "Out",
    source: "ESPN injury report",
    dated: "2026-11-01 08:00",
  };
}

beforeEach(() => {
  vi.mocked(api.getGamePlayers).mockReset();
  vi.mocked(api.getGameOutPlayers).mockReset();
});
afterEach(() => vi.restoreAllMocks());

describe("buildTopCalls — the cap is a ceiling", () => {
  it("keeps at most three rows per category and never a fourth", () => {
    const props = [
      prop("1", "A", "points", 30),
      prop("2", "B", "points", 29),
      prop("3", "C", "points", 28),
      prop("4", "D", "points", 27),
      prop("5", "E", "points", 26),
    ];
    const { categories } = buildTopCalls(props, []);
    const points = categories.find((c) => c.category === "Points")!;
    expect(points.rows).toHaveLength(3);
    // The three KEPT are the three highest: the cap truncates, it does not
    // shuffle in a fourth to pad the list.
    expect(points.rows.map((r) => r.name)).toEqual(["A", "B", "C"]);
  });

  it("shows fewer when fewer exist, and pads nothing", () => {
    const props = [prop("1", "A", "threes", 3)];
    const { categories } = buildTopCalls(props, []);
    const threes = categories.find((c) => c.category === "Threes")!;
    expect(threes.rows).toHaveLength(1);
  });

  it("ranks by the model's own number, descending", () => {
    const props = [prop("1", "Low", "assists", 4), prop("2", "High", "assists", 11)];
    const { categories } = buildTopCalls(props, []);
    const assists = categories.find((c) => c.category === "Assists")!;
    expect(assists.rows.map((r) => r.name)).toEqual(["High", "Low"]);
  });
});

describe("buildTopCalls — one category per list", () => {
  it("never files a rebounds row under a points heading", () => {
    const props = [
      prop("1", "Rebounder", "rebounds", 14),
      prop("2", "Scorer", "points", 31),
      prop("3", "Passer", "assists", 9),
      prop("4", "Shooter", "threes", 4),
    ];
    const { categories } = buildTopCalls(props, []);
    for (const { category, rows } of categories) {
      for (const row of rows) {
        expect(row.detail.toLowerCase()).toContain(category.toLowerCase());
      }
    }
    // And the four categories are the four that exist, each holding its own row.
    expect(categories.map((c) => c.category)).toEqual(["Points", "Rebounds", "Assists", "Threes"]);
    expect(categories.find((c) => c.category === "Points")!.rows.map((r) => r.name)).toEqual(["Scorer"]);
  });

  it("drops a category nobody produced rather than inventing an empty one", () => {
    const { categories } = buildTopCalls([prop("1", "Scorer", "points", 31)], []);
    expect(categories.map((c) => c.category)).toEqual(["Points"]);
  });
});

describe("buildTopCalls — nothing here is a probability", () => {
  it("marks every row a projection and says so on the row", () => {
    const { categories } = buildTopCalls([prop("1", "Scorer", "points", 31, 4.2)], []);
    const row = categories[0].rows[0];
    expect(row.kind).toBe("projection");
    expect(row.detail.toLowerCase()).toContain("projection");
    expect(row.detail.toLowerCase()).toContain("not a probability");
  });

  it("claims no price, no edge and no guarantee on any row", () => {
    const { categories } = buildTopCalls(
      [prop("1", "Scorer", "points", 31, 4.2), prop("2", "Shooter", "threes", 4, null)],
      [],
    );
    const words = categories.flatMap((c) => c.rows.flatMap((r) => [r.detail, r.provenance])).join(" ").toLowerCase();
    for (const banned of ["lock", "guaranteed", "best bet", "edge", "value", "odds", "moneyline"]) {
      expect(words, `banned wording in a row: ${banned}`).not.toContain(banned);
    }
  });
});

describe("buildTopCalls — provenance is honest about its unit", () => {
  it("states the ± MAE and that no graded per-player record exists", () => {
    const { categories } = buildTopCalls([prop("1", "Scorer", "points", 31, 4.2)], []);
    const row = categories[0].rows[0];
    expect(row.margin).toBe(4.2);
    expect(row.provenance.toLowerCase()).toContain("mae");
    expect(row.provenance.toLowerCase()).toContain("no graded per-player record");
  });

  it("says there is no error estimate rather than drawing ± 0 when mae is null", () => {
    const { categories } = buildTopCalls([prop("1", "Shooter", "threes", 4, null)], []);
    const row = categories[0].rows[0];
    // `undefined`, NOT 0: PicksList renders undefined as the honest sentence and
    // would render 0 as "± 0.0", which is a claim the model never missed.
    expect(row.margin).toBeUndefined();
    expect(row.provenance.toLowerCase()).toContain("no error estimate yet");
  });

  it("never borrows another stat's error: each row carries its own", () => {
    const { categories } = buildTopCalls(
      [prop("1", "Scorer", "points", 31, 4.2), prop("2", "Rebounder", "rebounds", 14, 2.1)],
      [],
    );
    const points = categories.find((c) => c.category === "Points")!.rows[0];
    const rebounds = categories.find((c) => c.category === "Rebounds")!.rows[0];
    expect(points.margin).toBe(4.2);
    expect(rebounds.margin).toBe(2.1);
  });
});

describe("buildTopCalls — an out player leaves the ranking entirely", () => {
  // The backend already withholds an out player's rows. This asserts the rule is
  // enforced by the page as well, on the strength of the out feed ALONE, so a
  // props feed that still carries him cannot put him back in a ranking.
  const props = [
    prop("1", "Healthy", "points", 31, 4.2),
    prop("2", "Absent", "points", 35, 3.1),
    prop("2", "Absent", "rebounds", 12, 2.1),
    prop("2", "Absent", "assists", 10, 2.6),
    prop("2", "Absent", "threes", 5, 1.2),
  ];
  const out = [outRow("2", "Absent")];

  it("keeps him out of every list", () => {
    const { categories } = buildTopCalls(props, out);
    for (const { rows } of categories) {
      for (const row of rows) expect(row.name).not.toBe("Absent");
    }
  });

  it("gives him exactly one out entry, not one per stat row", () => {
    const { out: shown } = buildTopCalls(props, out);
    expect(shown).toHaveLength(1);
    expect(shown[0]).toMatchObject({ name: "Absent", source: "ESPN injury report", dated: "2026-11-01 08:00" });
  });

  it("collapses a duplicate out feed to a single mention", () => {
    const { out: shown } = buildTopCalls(props, [outRow("2", "Absent"), outRow("2", "Absent")]);
    expect(shown).toHaveLength(1);
  });

  it("backfills with the next healthy player rather than showing an out one", () => {
    const more = [...props, prop("3", "Next", "points", 30, 4.0)];
    const { categories } = buildTopCalls(more, out);
    const points = categories.find((c) => c.category === "Points")!;
    expect(points.rows.map((r) => r.name)).toEqual(["Healthy", "Next"]);
  });
});

describe("TopCalls — what the reader actually sees", () => {
  it("titles the list 'Model's top calls'", () => {
    render(<TopCalls props={[prop("1", "Scorer", "points", 31, 4.2)]} out={[]} />);
    expect(screen.getByTestId("picks-title")).toHaveTextContent("Model's top calls");
  });

  it("renders the four category lists with their ± on the page", () => {
    render(
      <TopCalls
        props={[
          prop("1", "Scorer", "points", 31, 4.2),
          prop("2", "Rebounder", "rebounds", 14, 2.1),
          prop("3", "Passer", "assists", 9, 2.6),
          prop("4", "Shooter", "threes", 4, null),
        ]}
        out={[]}
      />,
    );
    const headings = screen.getAllByTestId("picks-category-heading").map((h) => h.textContent);
    expect(headings).toEqual(["Points", "Rebounds", "Assists", "Threes"]);
    // The points row's ±, and the threes row's honest absence of one.
    expect(screen.getByText("± 4.2")).toBeInTheDocument();
    expect(screen.getByText("no error estimate yet")).toBeInTheDocument();
    expect(screen.queryByText("± 0.0")).not.toBeInTheDocument();
  });

  it("renders every row as a projection, never a probability bar", () => {
    render(<TopCalls props={[prop("1", "Scorer", "points", 31, 4.2)]} out={[]} />);
    const rows = screen.getAllByTestId("picks-row");
    expect(rows).toHaveLength(1);
    expect(rows[0]).toHaveAttribute("data-kind", "projection");
    expect(screen.queryByTestId("prob-bar")).not.toBeInTheDocument();
  });

  it("shows an out player exactly once below the lists, and never in one", () => {
    const { container } = render(
      <TopCalls
        props={[
          prop("1", "Healthy", "points", 31, 4.2),
          prop("2", "Absent", "points", 35, 3.1),
          prop("2", "Absent", "rebounds", 12, 2.1),
        ]}
        out={[outRow("2", "Absent")]}
      />,
    );
    const lists = screen.getAllByTestId("picks-category");
    for (const list of lists) {
      expect(within(list).queryByText(/Absent/)).not.toBeInTheDocument();
    }
    const outLine = screen.getByTestId("picks-out");
    expect(outLine).toHaveTextContent("Absent");
    expect(outLine).toHaveTextContent("ESPN injury report");
    expect(outLine).toHaveTextContent("2026-11-01 08:00");
    // Once. Not once per category, and not as a row.
    expect(container.textContent!.match(/Absent/g)).toHaveLength(1);
  });

  it("renders nothing at all rather than an empty shell", () => {
    const { container } = render(<TopCalls props={[]} out={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders only the out line when the whole field is out", () => {
    render(<TopCalls props={[prop("2", "Absent", "points", 35, 3.1)]} out={[outRow("2", "Absent")]} />);
    expect(screen.queryByTestId("picks-row")).not.toBeInTheDocument();
    expect(screen.getByTestId("picks-out")).toHaveTextContent("Absent");
  });
});
