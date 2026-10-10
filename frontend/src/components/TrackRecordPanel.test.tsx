import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import TrackRecordPanel from "./TrackRecordPanel";
import { wilson } from "./trackRecordViews";
import { api, type TrackRecord, type TrackRecordPick, type TrackRecordWeek, type VsMarket } from "../api/client";

vi.mock("../api/client", () => ({ api: { getTrackRecord: vi.fn(), getVsMarket: vi.fn() } }));

const wk = (week_start: string, n: number, correct: number, tracked = true): TrackRecordWeek => ({
  week_start, n, correct, hit_rate: n ? correct / n : null, tracked,
});
const pick = (over: Partial<TrackRecordPick>): TrackRecordPick => ({
  game_id: "g1", market: "h2h", pick: "BOS", actual: "BOS", hit: true, made_before_tip: true,
  created_at: "2026-10-05T03:00:00Z", counted: true, gameday: "2026-10-08", point: null, ...over,
});
type Row = TrackRecord & { confidence_buckets?: unknown };
const row = (over: Partial<Row> & { market: string }): Row => ({
  total_predictions: 0, correct_predictions: 0, hit_rate: null, n_rebuilt: 0, n_push: 0, settled: true, weekly: [], ...over,
});
const vs = (over: Partial<VsMarket> = {}): VsMarket => ({
  market: "h2h", n: 10, mean_model_probability: 0.6, mean_market_probability: 0.58, mean_edge_points: 2,
  disagreement_n: 2, disagreement_hit_rate: 0.5, disagreement_game_ids: ["g1", "g2"], weekly: [],
  scope: { population: "x", weekly_from: null, weekly_through: null, n_games_total: 10, n_games_in_weekly: 10, n_games_outside_weekly: 0 },
  method: {}, ...over,
});

// The live shape: the winner pick is mostly rebuilt after tip-off, with a small counted slice.
const LIVE: Row[] = [
  row({
    market: "game_outcome", total_predictions: 1389, correct_predictions: 1182, hit_rate: 0.851, n_rebuilt: 1365,
    pre_tip: { total_predictions: 24, correct_predictions: 11, hit_rate: 0.458, weekly: [wk("2026-09-21", 0, 0, false), wk("2026-09-28", 3, 2), wk("2026-10-05", 21, 9)] },
    weekly: [],
  }),
  row({
    market: "h2h", total_predictions: 10, correct_predictions: 5, hit_rate: 0.5, n_rebuilt: 0,
    pre_tip: { total_predictions: 10, correct_predictions: 5, hit_rate: 0.5 },
    per_pick: [pick({ game_id: "g1" }), pick({ game_id: "g2", hit: false, pick: "ATL", actual: "MEM" }), pick({ game_id: "old", counted: false })],
    confidence_buckets: [
      { bucket: "50-60%", total_predictions: 4, correct_predictions: 2, hit_rate: 0.5 },
      { bucket: "60-70%", total_predictions: 3, correct_predictions: 2, hit_rate: 0.667 },
      { bucket: "70%+", total_predictions: 3, correct_predictions: 1, hit_rate: 0.333 },
    ],
  }),
  row({ market: "player_props", settled: false, total_predictions: 121148 }),
];

beforeEach(() => {
  vi.mocked(api.getTrackRecord).mockResolvedValue(LIVE as TrackRecord[]);
  vi.mocked(api.getVsMarket).mockResolvedValue(vs());
});

describe("TrackRecordPanel", () => {
  it("leads with the counted pre-tip record, not the rebuilt 85%", async () => {
    render(<TrackRecordPanel />);
    const lead = await screen.findByTestId("tally-game_outcome");
    expect(lead).toHaveTextContent("11 of 24");
    // The blended 85% (counted + rebuilt) is shown nowhere: the headline is the pre-tip slice only.
    expect(document.body).not.toHaveTextContent("85%");
    expect(lead).toHaveTextContent("46%");
  });

  it("says plainly that a small sample is early", async () => {
    render(<TrackRecordPanel />);
    expect(await screen.findByTestId("early-sample")).toHaveTextContent("24 counted picks");
  });

  it("keeps rebuilt-after-tip rows in a separate block that never feeds the headline", async () => {
    render(<TrackRecordPanel />);
    const block = await screen.findByTestId("backtest-game_outcome");
    expect(block).toHaveTextContent("1,171 of 1,365");
    expect(screen.getByRole("heading", { name: /backtest, not counted/i })).toBeInTheDocument();
  });

  it("draws one cell per tracked week and starts at the first graded week", async () => {
    render(<TrackRecordPanel />);
    await waitFor(() => expect(screen.getAllByTestId("week-cell")).toHaveLength(2));
  });

  it("plots a calibration point per graded bucket", async () => {
    render(<TrackRecordPanel />);
    await waitFor(() => expect(screen.getAllByTestId("calibration-point")).toHaveLength(3));
  });

  it("lists the disagreement games with how each ended, and only counted picks", async () => {
    render(<TrackRecordPanel />);
    const rows = await screen.findAllByTestId("disagreement");
    expect(rows).toHaveLength(2);
    expect(rows[1]).toHaveTextContent("ATL");
    expect(rows[1]).toHaveTextContent("Wrong");
  });

  it("lists the latest counted picks and leaves out the one that lost the earliest-pick contest", async () => {
    render(<TrackRecordPanel />);
    const rows = await screen.findAllByTestId("recent-pick");
    expect(rows).toHaveLength(2);
  });

  it("never prints a rate for a market the backend cannot judge", async () => {
    render(<TrackRecordPanel />);
    await screen.findByTestId("tally-game_outcome");
    expect(screen.queryByTestId("tally-player_props")).toBeNull();
  });

  it("falls back to a plain message when nothing counted has graded", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([row({ market: "h2h" })] as TrackRecord[]);
    render(<TrackRecordPanel />);
    expect(await screen.findByText(/no pick made before tip-off has been graded/i)).toBeInTheDocument();
  });

  it("keeps a broken odds comparison from taking the record down", async () => {
    vi.mocked(api.getVsMarket).mockRejectedValue(new Error("x"));
    render(<TrackRecordPanel />);
    expect(await screen.findByTestId("tally-game_outcome")).toBeInTheDocument();
    expect(await screen.findByRole("alert")).toHaveTextContent(/comparison with the odds/i);
  });

  it("shows the error state with a retry when the record cannot load", async () => {
    vi.mocked(api.getTrackRecord).mockRejectedValue(new Error("x"));
    render(<TrackRecordPanel />);
    expect(await screen.findByText(/couldn't load the track record/i)).toBeInTheDocument();
  });
});

describe("wilson", () => {
  it("is a wide range for a small sample and null for none", () => {
    const [lo, hi] = wilson(11, 24)!;
    expect(lo).toBeGreaterThan(0.25);
    expect(lo).toBeLessThan(0.3);
    expect(hi).toBeGreaterThan(0.64);
    expect(hi).toBeLessThan(0.68);
    expect(wilson(0, 0)).toBeNull();
  });
});
