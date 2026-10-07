import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import TrackRecordPanel from "./TrackRecordPanel";
import { api, type TrackRecord, type TrackRecordWeek, type VsMarket, type VsMarketWeek } from "../api/client";

vi.mock("../api/client", () => ({ api: { getTrackRecord: vi.fn(), getVsMarket: vi.fn() } }));

const WEEKLY = "Accuracy by week, per market";

/** A track-record row in the shape /hub/track-record actually sends. */
function row(over: Partial<TrackRecord> & { market: string }): TrackRecord {
  return {
    total_predictions: 0,
    correct_predictions: 0,
    hit_rate: null,
    n_rebuilt: 0,
    n_push: 0,
    settled: true,
    weekly: [],
    ...over,
  };
}

/** One week of a market's weekly list: a gap week is tracked=false, hit_rate null. */
function week(over: Partial<TrackRecordWeek> & { week_start: string }): TrackRecordWeek {
  return { n: 0, correct: 0, hit_rate: null, tracked: false, ...over };
}

function vsWeek(over: Partial<VsMarketWeek> & { week_start: string }): VsMarketWeek {
  return { tracked: true, n: 0, mean_edge_points: null, disagreement_n: 0, disagreement_hit_rate: null, ...over };
}

/** A /hub/vs-market payload with an empty (n = 0) body by default. */
function vsMarket(over: Partial<VsMarket> = {}): VsMarket {
  return {
    market: "h2h",
    n: 0,
    mean_model_probability: null,
    mean_market_probability: null,
    mean_edge_points: null,
    disagreement_n: 0,
    disagreement_hit_rate: null,
    disagreement_game_ids: [],
    weekly: [],
    scope: {
      population: "finished games with a pre-tip moneyline price",
      weekly_from: null,
      weekly_through: null,
      n_games_total: 0,
      n_games_in_weekly: 0,
      n_games_outside_weekly: 0,
    },
    method: {},
    ...over,
  };
}

/** The data rows of the table whose sr-only caption reads `caption`. */
function tableRows(caption: string): HTMLElement[] {
  const table = screen.getByText(caption).closest("table");
  if (!table) throw new Error(`no table carries the caption "${caption}"`);
  return within(table).getAllByRole("row").slice(1);
}

function rowWith(caption: string, needle: string): HTMLElement {
  const found = tableRows(caption).find((r) => (r.textContent ?? "").includes(needle));
  if (!found) throw new Error(`no row of "${caption}" contains "${needle}"`);
  return found;
}

/** The coloured fill inside an AccuracyBar: the width IS the claim under test. */
const fillOf = (bar: HTMLElement) => bar.firstElementChild as HTMLElement;

beforeEach(() => {
  vi.mocked(api.getTrackRecord).mockResolvedValue([]);
  vi.mocked(api.getVsMarket).mockResolvedValue(vsMarket());
});
afterEach(() => vi.restoreAllMocks());

describe("TrackRecordPanel", () => {
  it("shows no section for data the site does not have", async () => {
    // The "Projected final standings" section shipped as a heading and a blurb
    // promising projected win totals, above a body reading "endpoint not yet
    // implemented" -- a promise of a number the page then withheld. It is gone.
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      row({ market: "h2h", total_predictions: 100, correct_predictions: 58, hit_rate: 0.58 }),
    ]);
    vi.mocked(api.getVsMarket).mockResolvedValue(vsMarket());

    render(<TrackRecordPanel />);
    await screen.findByText("Moneyline vs the market accuracy");

    expect(screen.queryByText("Projected final standings")).toBeNull();
    expect(document.body.textContent).not.toMatch(/not yet implemented/i);
    expect(document.body.textContent).not.toMatch(/projected to 82 games/i);
  });

  it("shows a settled market's rate, with the record behind it", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      row({ market: "h2h", total_predictions: 100, correct_predictions: 58, hit_rate: 0.58 }),
    ]);

    render(<TrackRecordPanel />);

    expect(await screen.findByText("Moneyline vs the market accuracy")).toBeInTheDocument();
    expect(screen.getByText("58%")).toBeInTheDocument();
    // The n travels with the rate: never a bare percentage.
    expect(screen.getByText("58/100")).toBeInTheDocument();
  });

  it("counts every recorded pick, says how many were made after tip-off, and shows the pre-tip subset", async () => {
    // The old test was "says pre-tip picks only, and counts rebuilt finals and
    // pushes out loud", and asserted the strings "Only picks made before
    // tip-off count" and "1,200 finals had only a pick rebuilt after tip-off, so
    // they are left out." Both are gone: nothing is withheld any more, and a
    // figure that is counted is not described as left out.
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      row({
        market: "game_outcome", total_predictions: 1240, correct_predictions: 806, hit_rate: 0.65,
        n_rebuilt: 1200, n_push: 2, n_pre_tip: 40,
        pre_tip: { total_predictions: 40, correct_predictions: 26, hit_rate: 0.65 },
      }),
    ]);
    render(<TrackRecordPanel />);

    expect(await screen.findByText("Winner pick accuracy")).toBeInTheDocument();
    expect(screen.getByText(/Every recorded pick counts, whenever it was made/)).toBeInTheDocument();
    // Disclosure, not exclusion: the count is named as part of the headline.
    expect(screen.getByTestId("timing-note")).toHaveTextContent(
      "1,200 of these 1,240 picks were recorded at or after their own tip-off",
    );
    // The old copy is gone, both halves of it.
    expect(screen.queryByText(/Only picks made before tip-off count/)).toBeNull();
    expect(screen.queryByText(/had only a pick rebuilt after tip-off/)).toBeNull();
    expect(screen.queryByText(/not counted here/)).toBeNull();
    // A push is still left out of the rate, not scored as a miss.
    expect(screen.getByText(/2 picks landed on the line and were left out of the rate, not scored as a miss\./)).toBeInTheDocument();
    // And the secondary figure carries its own n, beside the headline.
    const PRE_TIP = "Accuracy on picks made before tip-off";
    await screen.findByText(PRE_TIP);
    const preTipRow = rowWith(PRE_TIP, "Winner pick");
    expect(preTipRow).toHaveTextContent("40");
    expect(preTipRow).toHaveTextContent("26/40");
    expect(preTipRow).toHaveTextContent("65%");
  });

  it("says nothing was made after tip-off when the two figures are the same picks", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      row({
        market: "game_outcome", total_predictions: 40, correct_predictions: 26, hit_rate: 0.65,
        n_rebuilt: 0, n_push: 0, n_pre_tip: 40,
        pre_tip: { total_predictions: 40, correct_predictions: 26, hit_rate: 0.65 },
      }),
    ]);
    render(<TrackRecordPanel />);

    expect(await screen.findByTestId("timing-note")).toHaveTextContent(
      "Every pick in this record was made before its game tipped off",
    );
  });

  it("never prints a rate for a market with no pre-tip figure to compare", async () => {
    // A settled market whose pre-tip subset graded nothing reads as a dash,
    // never 0% -- "never measured" is not "measured at zero".
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      row({
        market: "h2h", total_predictions: 300, correct_predictions: 150, hit_rate: 0.5,
        n_rebuilt: 300, n_pre_tip: 0,
        pre_tip: { total_predictions: 0, correct_predictions: 0, hit_rate: null },
      }),
    ]);
    render(<TrackRecordPanel />);

    const PRE_TIP = "Accuracy on picks made before tip-off";
    await screen.findByText(PRE_TIP);
    const preTipRow = rowWith(PRE_TIP, "Moneyline");
    expect(preTipRow).toHaveTextContent("—");
    expect(preTipRow.textContent).not.toMatch(/%/);
  });

  it("never prints a rate for a market the backend has not settled", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      row({ market: "spread", total_predictions: 12, correct_predictions: 0, hit_rate: 0, settled: false }),
      row({ market: "game_outcome", total_predictions: 40, correct_predictions: 26, hit_rate: 0.65 }),
    ]);
    render(<TrackRecordPanel />);

    expect(await screen.findByText("Winner pick accuracy")).toBeInTheDocument();
    expect(screen.getByText("65%")).toBeInTheDocument();
    // The stored 0.0 hit_rate of an ungraded market must not surface as 0%.
    expect(screen.queryByText("Spread accuracy")).not.toBeInTheDocument();
    expect(screen.queryByText("0%")).not.toBeInTheDocument();
    const note = screen.getByTestId("unsettled-note");
    expect(note).toHaveTextContent("Spread: 12 stored rows, not settled yet");
    expect(note).toHaveTextContent("No rate is shown for them, because none has been measured.");
  });

  it("shows no rate at all when the payload carries no settled flag", async () => {
    // Older payloads predate `settled`; absent means unmeasured, not 0%.
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      { market: "h2h", total_predictions: 9, correct_predictions: 5, hit_rate: 0.555 },
    ]);
    render(<TrackRecordPanel />);

    expect(await screen.findByTestId("unsettled-note")).toBeInTheDocument();
    expect(screen.getByText(/9 stored rows, not settled yet/)).toBeInTheDocument();
    expect(screen.queryByText(/\d+%/)).not.toBeInTheDocument();
  });

  it("reads as a dash, never 0%, when a settled market has graded nothing", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([row({ market: "total" })]);
    render(<TrackRecordPanel />);

    expect(await screen.findByText("Total points accuracy")).toBeInTheDocument();
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
    expect(screen.queryByText("0%")).not.toBeInTheDocument();
    expect(screen.queryByText("0/0")).not.toBeInTheDocument();
  });

  it("shows a week with nothing graded as Not tracked, never as 0%", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      row({
        market: "game_outcome",
        total_predictions: 4,
        correct_predictions: 3,
        hit_rate: 0.75,
        weekly: [
          week({ week_start: "2026-09-21", n: 4, correct: 3, hit_rate: 0.75, tracked: true }),
          week({ week_start: "2026-09-28", n: 0, correct: 0, hit_rate: null, tracked: false }),
        ],
      }),
      row({
        market: "spread",
        total_predictions: 1,
        correct_predictions: 1,
        hit_rate: 1,
        weekly: [
          week({ week_start: "2026-09-21", n: 1, correct: 1, hit_rate: 1, tracked: true }),
          week({ week_start: "2026-09-28", n: 0, correct: 0, hit_rate: null, tracked: false }),
        ],
      }),
    ]);
    render(<TrackRecordPanel />);
    await screen.findByText(WEEKLY);

    const gap = rowWith(WEEKLY, "Not tracked");
    // One gap per market, and not a single percentage in the row.
    expect(within(gap).getAllByText("Not tracked")).toHaveLength(2);
    expect(gap.textContent).not.toMatch(/%/);

    const graded = tableRows(WEEKLY)[0];
    expect(graded.textContent).toMatch(/Mon 21 Sep/);
    expect(graded.textContent).toMatch(/75% \(4\)/);
    expect(graded.textContent).toMatch(/99% \(1\)/);
    expect(screen.queryByText("0%")).not.toBeInTheDocument();
  });

  it("shows a dash for a week that had picks but nothing measurable", async () => {
    // Two picks, both pushes: n > 0 with hit_rate null.
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      row({
        market: "game_outcome",
        total_predictions: 2,
        hit_rate: null,
        weekly: [week({ week_start: "2026-09-21", n: 2, correct: 0, hit_rate: null, tracked: true })],
      }),
    ]);
    render(<TrackRecordPanel />);
    await screen.findByText(WEEKLY);

    const only = tableRows(WEEKLY)[0];
    expect(only.textContent).toMatch(/— \(2\)/);
    expect(only.textContent).not.toMatch(/0%/);
  });

  it("scales the bar to accuracy alone, never to how many games the week held", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      row({
        market: "game_outcome",
        total_predictions: 5,
        correct_predictions: 3,
        hit_rate: 0.75,
        weekly: [
          // One perfect game: full bar. Four games at 50%: half bar.
          week({ week_start: "2026-09-21", n: 1, correct: 1, hit_rate: 1, tracked: true }),
          week({ week_start: "2026-09-28", n: 4, correct: 2, hit_rate: 0.5, tracked: true }),
        ],
      }),
    ]);
    render(<TrackRecordPanel />);
    await screen.findByText(WEEKLY);

    const bars = screen.getAllByTestId("week-bar");
    expect(bars).toHaveLength(2);
    expect(fillOf(bars[0]).style.width).toBe("100%");
    expect(fillOf(bars[1]).style.width).toBe("50%");
    // The headline card is on the same accuracy scale.
    expect(fillOf(screen.getByTestId("accuracy-bar-game_outcome")).style.width).toBe("75%");
    // Every bar carries the 50% line as a mark, not as a colour cue alone.
    expect(screen.getAllByTestId("accuracy-50-marker").length).toBeGreaterThan(0);
  });

  it("never scales by games played or falls back to a hard-coded settled list", () => {
    // Vite rewrites `new URL(x, import.meta.url)` into an HTTP asset URL, so
    // resolve from the runner's directory instead.
    const panelFile = ["src/components/TrackRecordPanel.tsx", "frontend/src/components/TrackRecordPanel.tsx"]
      .map((p) => resolve(process.cwd(), p))
      .find((p) => existsSync(p));
    expect(panelFile, "TrackRecordPanel.tsx not found from the test runner's cwd").toBeTruthy();
    const src = readFileSync(panelFile as string, "utf8");
    // The bar is the accuracy scale: no volume denominator may reach it.
    expect(src).not.toMatch(/max_games/);
    expect(src).not.toMatch(/n_games\w*\s*\/(?!\/)/);
    expect(src).not.toMatch(/total_predictions\s*\//);
    // Settlement comes from the payload's flag, not from a list in the page.
    expect(src).not.toMatch(/\bSETTLED\b/);
    expect(src).toMatch(/row\.settled/);
  });

  it("prints the backend's method sentences verbatim, including unknown keys", async () => {
    const profitSentence =
      "This is agreement with a price, not a profit claim. No figure here is a return, a yield, a stake or a cent. " +
      "We do not publish profit or ROI figures, and this comparison does not become one by being labelled 'edge'.";
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      row({ market: "game_outcome", total_predictions: 4, correct_predictions: 3, hit_rate: 0.75 }),
    ]);
    vi.mocked(api.getVsMarket).mockResolvedValue(
      vsMarket({
        n: 6,
        mean_model_probability: 0.57,
        mean_market_probability: 0.54,
        mean_edge_points: 3.2,
        disagreement_n: 2,
        disagreement_hit_rate: 0.5,
        disagreement_game_ids: ["g1", "g2"],
        method: {
          not_a_profit_claim: profitSentence,
          closing_line: "The closing line sentence, owned by the backend.",
        },
      }),
    );
    render(<TrackRecordPanel />);

    const block = await screen.findByTestId("method-block");
    // Verbatim: the backend's wording, whole, not paraphrased or trimmed.
    expect(block).toHaveTextContent(profitSentence);
    expect(screen.getByText("What this is not")).toBeInTheDocument();
    // A key the label map does not know still reaches the page.
    expect(screen.getByText("closing line")).toBeInTheDocument();
    expect(block).toHaveTextContent("The closing line sentence, owned by the backend.");
    expect(screen.queryByText(/profit or ROI of/i)).not.toBeInTheDocument();
  });

  it("says who is in the number when the weekly table cannot add up to it", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      row({ market: "game_outcome", total_predictions: 4, correct_predictions: 3, hit_rate: 0.75 }),
    ]);
    vi.mocked(api.getVsMarket).mockResolvedValue(
      vsMarket({
        n: 10,
        mean_model_probability: 0.57,
        mean_market_probability: 0.54,
        mean_edge_points: 3.2,
        disagreement_n: 4,
        disagreement_hit_rate: 0.5,
        disagreement_game_ids: ["g1", "g2", "g3", "g4"],
        weekly: [
          vsWeek({ week_start: "2026-09-21", n: 5, mean_edge_points: 2.5, disagreement_n: 2, disagreement_hit_rate: 0.5 }),
          vsWeek({ week_start: "2026-09-28", n: 0, tracked: false }),
        ],
        scope: {
          population: "finished games with a pre-tip moneyline price",
          weekly_from: "2026-09-21",
          weekly_through: "2026-09-28",
          n_games_total: 10,
          n_games_in_weekly: 8,
          n_games_outside_weekly: 2,
        },
      }),
    );
    render(<TrackRecordPanel />);

    const note = await screen.findByTestId("scope-note");
    expect(note).toHaveTextContent("cover 10 games");
    expect(note).toHaveTextContent("finished games with a pre-tip moneyline price");
    expect(note).toHaveTextContent("accounts for 8 of them");
    expect(note).toHaveTextContent("2 games in the headline fall outside that window");
    expect(note).toHaveTextContent("the table does not add up to the count above");
    // The cohort carries its n too.
    expect(screen.getByText(/Hit rate over 4 games where it disagreed with the price\./)).toBeInTheDocument();

    // A week with nothing compared reads as its own gaps, never as 0 pt or 0%.
    const vsRows = tableRows("Model against the price, by week");
    expect(vsRows).toHaveLength(2);
    expect(vsRows[0].textContent).toMatch(/5/);
    expect(vsRows[1].textContent).toContain("Not compared");
    expect(vsRows[1].textContent).toContain("No disagreement");
    expect(vsRows[1].textContent).toMatch(/—/);
    expect(vsRows[1].textContent).not.toMatch(/%/);
  });

  it("withholds the scope note when the scope's own identity does not hold", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      row({ market: "game_outcome", total_predictions: 4, correct_predictions: 3, hit_rate: 0.75 }),
    ]);
    vi.mocked(api.getVsMarket).mockResolvedValue(
      vsMarket({
        n: 10,
        mean_model_probability: 0.57,
        mean_market_probability: 0.54,
        mean_edge_points: 3.2,
        disagreement_n: 4,
        disagreement_hit_rate: 0.5,
        disagreement_game_ids: [],
        scope: {
          population: "finished games with a pre-tip moneyline price",
          weekly_from: "2026-09-21",
          weekly_through: "2026-09-28",
          n_games_total: 10,
          n_games_in_weekly: 8,
          n_games_outside_weekly: 1, // 8 + 1 !== 10: a claim we cannot back
        },
      }),
    );
    render(<TrackRecordPanel />);

    await screen.findByText("Games compared");
    expect(screen.queryByTestId("scope-note")).not.toBeInTheDocument();
    // Withholding the note hides nothing else: the figures still render.
    expect(screen.getByText("Vs the market")).toBeInTheDocument();
  });

  it("keeps a broken comparison from taking the record down, and retries it alone", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      row({ market: "game_outcome", total_predictions: 40, correct_predictions: 26, hit_rate: 0.65 }),
    ]);
    vi.mocked(api.getVsMarket).mockRejectedValue(new Error("boom"));
    render(<TrackRecordPanel />);

    expect(await screen.findByText("Winner pick accuracy")).toBeInTheDocument();
    expect(screen.getByText("65%")).toBeInTheDocument();

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("We couldn't load the model-versus-market comparison.");

    vi.mocked(api.getVsMarket).mockResolvedValue(vsMarket({ n: 3, disagreement_n: 0 }));
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));

    await waitFor(() => expect(screen.getByText("Games compared")).toBeInTheDocument());
    expect(vi.mocked(api.getVsMarket).mock.calls).toHaveLength(2);
    // The record above it was never thrown away by the retry.
    expect(screen.getByText("65%")).toBeInTheDocument();
  });

  it("explains an empty disagreement cohort instead of inventing a 50% hit rate", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      row({ market: "game_outcome", total_predictions: 4, correct_predictions: 3, hit_rate: 0.75 }),
    ]);
    vi.mocked(api.getVsMarket).mockResolvedValue(
      vsMarket({ n: 6, mean_model_probability: 0.5, mean_market_probability: 0.5, mean_edge_points: 0, disagreement_n: 0, disagreement_hit_rate: null }),
    );
    render(<TrackRecordPanel />);

    expect(await screen.findByText(/Hit rate over 0 games where it disagreed with the price\./)).toBeInTheDocument();
    expect(screen.getByText(/A 50\/50 price favours nobody, so those games are compared but left out of this cohort\./)).toBeInTheDocument();
    expect(screen.queryByText("0%")).not.toBeInTheDocument();
  });

  it("says plainly when no finished game has a price to compare with", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      row({ market: "game_outcome", total_predictions: 4, correct_predictions: 3, hit_rate: 0.75 }),
    ]);
    render(<TrackRecordPanel />);

    expect(await screen.findByText("Vs the market")).toBeInTheDocument();
    expect(screen.getByText(/No finished game has a pre-tip moneyline price to compare with yet\./)).toBeInTheDocument();
    expect(screen.queryByTestId("scope-note")).not.toBeInTheDocument();
  });

  it("marks the 50% break-even line on each hit-rate bar", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      row({ market: "h2h", total_predictions: 100, correct_predictions: 58, hit_rate: 0.58 }),
      row({ market: "spread", total_predictions: 100, correct_predictions: 51, hit_rate: 0.51 }),
    ]);

    render(<TrackRecordPanel />);

    await waitFor(() => expect(screen.getAllByTestId("accuracy-50-marker")).toHaveLength(2));
  });

  it("renders a dash instead of a misleading 0% for the player-props row", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      row({ market: "player_props", total_predictions: 200, correct_predictions: 0, hit_rate: null }),
    ]);

    render(<TrackRecordPanel />);

    await waitFor(() => expect(screen.getByText("Player props accuracy")).toBeInTheDocument());
    expect(screen.getByText("—")).toBeInTheDocument();
    expect(screen.queryByText("0%")).not.toBeInTheDocument();
  });

  it("shows an empty state when no predictions are tracked yet", async () => {
    render(<TrackRecordPanel />);
    await waitFor(() => expect(screen.getByText(/no tracked predictions/i)).toBeInTheDocument());
  });
});
