import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import GameDetailModal from "./GameDetailModal";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: { getGameDetail: vi.fn(), getGamePlayers: vi.fn() },
}));

afterEach(() => {
  vi.restoreAllMocks();
});

const completedDetail = {
  game_id: "g2", game_date: "2026-01-05", home_team: "BOS", away_team: "MIA",
  completed: true, home_pts: 113, away_pts: 105,
  prediction: { home_win_probability: 0.62, predicted_margin: 3.5, predicted_total: 224.5 },
  markets: [], head_to_head: [], home_recent_form: [], away_recent_form: [],
};

const detail = {
  game_id: "g1", game_date: "2026-11-01", home_team: "BOS", away_team: "MIA",
  prediction: { home_win_probability: 0.62, predicted_margin: 3.5, predicted_total: 224.5 },
  completed: false, home_pts: null, away_pts: null,
  markets: [
    { market: "h2h", selection: "BOS", model_probability: 0.62, market_probability: 0.55, edge: 0.07, bookmaker: "DraftKings", american_odds: -130, point: null },
    { market: "spread", selection: "BOS", model_probability: 0.52, market_probability: 0.5, edge: 0.02, bookmaker: "DraftKings", american_odds: -110, point: -4.5 },
    { market: "total", selection: "over", model_probability: 0.48, market_probability: 0.5, edge: -0.02, bookmaker: "FanDuel", american_odds: -105, point: 224.5 },
  ],
  head_to_head: [
    { game_id: "g0", game_date: "2026-01-01", home_team: "BOS", away_team: "MIA", home_pts: 110, away_pts: 100 },
  ],
  home_recent_form: ["W", "L", "W"],
  away_recent_form: ["L", "L", "W"],
};

const players = [{ player_id: "203999", player_name: "Nikola Jokic", stat: "points", predicted_value: 27.5, actual_value: null }];

describe("GameDetailModal", () => {
  it("shows a loading state before data arrives", () => {
    vi.mocked(api.getGameDetail).mockReturnValue(new Promise(() => {}));
    vi.mocked(api.getGamePlayers).mockReturnValue(new Promise(() => {}));

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it("renders markets sorted by edge descending", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);

    const rows = await screen.findAllByTestId("market-row");
    expect(rows[0]).toHaveTextContent("h2h");
    expect(rows[1]).toHaveTextContent("spread");
  });

  it("renders player prop predictions", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);

    expect(await screen.findByText("points")).toBeInTheDocument();
    expect(screen.getByText("27.5")).toBeInTheDocument();
  });

  it("renders bookmaker and american odds for each market row", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);

    await screen.findAllByTestId("market-row");
    const bookmakers = screen.getAllByText(/DraftKings|FanDuel/);
    expect(bookmakers).toHaveLength(3);
    expect(screen.getByText(/-130/)).toBeInTheDocument();
  });

  it("renders the spread and total lines", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);

    await screen.findAllByTestId("market-row");
    expect(screen.getAllByText(/-4\.5/)).toHaveLength(1);
    expect(screen.getAllByText(/224\.5/)).toHaveLength(2);
  });

  it("renders head-to-head history", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);

    expect(await screen.findByText("Thu 1 Jan")).toBeInTheDocument();
    expect(screen.getByText(/MIA 100 – 110 BOS/)).toBeInTheDocument();
  });

  it("renders recent form for both teams", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);

    expect(await screen.findAllByTestId("form-badge")).toHaveLength(6);
  });

  it("shows the final score instead of a live prediction for a completed game", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue({
      ...detail, completed: true, home_pts: 108, away_pts: 101, markets: [], head_to_head: [],
    });
    vi.mocked(api.getGamePlayers).mockResolvedValue([]);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);

    expect(await screen.findByText(/final/i)).toBeInTheDocument();
    expect(screen.getByText("108")).toBeInTheDocument();
    expect(screen.getByText("101")).toBeInTheDocument();
  });

  it("calls onClose when the close button is clicked", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);
    const onClose = vi.fn();

    render(<GameDetailModal gameId="g1" onClose={onClose} />);
    await screen.findByRole("heading", { name: /Celtics/ });

    await userEvent.click(screen.getByRole("button", { name: /close/i }));
    expect(onClose).toHaveBeenCalled();
  });

  it("shows an error state when either fetch fails", async () => {
    vi.mocked(api.getGameDetail).mockRejectedValue(new Error("boom"));
    vi.mocked(api.getGamePlayers).mockResolvedValue([]);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);
    expect(await screen.findByText(/couldn.t load this game/i)).toBeInTheDocument();
  });

  it("closes when Escape is pressed", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);
    const onClose = vi.fn();

    render(<GameDetailModal gameId="g1" onClose={onClose} />);
    await screen.findByRole("heading", { name: /Celtics/ });

    await userEvent.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalled();
  });
});

it("shows a correct winner-call verdict when the favorite actually won", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(completedDetail);
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  expect(await screen.findByText("Called it ✓")).toBeInTheDocument();
  expect(screen.getByText("Pick before tip-off: BOS · 62%")).toBeInTheDocument();
});

it("shows an incorrect winner-call verdict when the underdog actually won", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue({ ...completedDetail, home_pts: 90, away_pts: 100 });
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  expect(await screen.findByText("Missed ✗")).toBeInTheDocument();
});

it("shows predicted vs actual margin with the absolute difference", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(completedDetail);
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  expect(await screen.findByText(/predicted margin/i)).toBeInTheDocument();
  expect(screen.getByText(/off by 4.5/i)).toBeInTheDocument();
});

it("shows predicted vs actual total with the absolute difference", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(completedDetail);
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  expect(await screen.findByText(/predicted total/i)).toBeInTheDocument();
  expect(screen.getByText(/off by 6.5/i)).toBeInTheDocument();
});

it("does not show a post-match verdict for an upcoming game", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(detail);
  vi.mocked(api.getGamePlayers).mockResolvedValue(players);

  render(<GameDetailModal gameId="g1" onClose={() => {}} />);

  await screen.findByRole("heading", { name: /Celtics/ });
  expect(screen.queryByTestId("post-match-verdict")).not.toBeInTheDocument();
});

const settledDetail = {
  ...completedDetail,
  markets: [
    { market: "h2h", selection: "BOS", model_probability: 0.62, market_probability: 0.55, edge: 0.07, bookmaker: "DraftKings", american_odds: -130, point: null },
    { market: "spread", selection: "MIA", model_probability: 0.4, market_probability: 0.45, edge: -0.05, bookmaker: "DraftKings", american_odds: 110, point: 4.5 },
    { market: "total", selection: "over", model_probability: 0.5, market_probability: 0.5, edge: 0.0, bookmaker: "FanDuel", american_odds: -105, point: 224.5 },
  ],
};

it("marks an h2h selection that matches the actual winner as a hit", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(settledDetail);
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  const rows = await screen.findAllByTestId("market-row");
  expect(rows[0]).toHaveTextContent("✓");
});

it("marks a spread selection that failed to cover as a miss", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(settledDetail);
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  const rows = await screen.findAllByTestId("market-row");
  const spreadRow = rows.find((r) => r.textContent?.includes("spread"));
  expect(spreadRow).toHaveTextContent("✗");
});

it("shows no verdict for a market row without a point value on an unsettled market", async () => {
  const noOddsDetail = { ...completedDetail, markets: [] };
  vi.mocked(api.getGameDetail).mockResolvedValue(noOddsDetail);
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  await screen.findByRole("heading", { name: /Celtics/ });
  expect(screen.queryAllByTestId("market-row")).toHaveLength(0);
});

it("shows predicted vs actual for a settled player prop", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(completedDetail);
  vi.mocked(api.getGamePlayers).mockResolvedValue([
    { player_id: "203999", player_name: "Nikola Jokic", stat: "points", predicted_value: 27.5, actual_value: 24.0 },
  ]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  expect(await screen.findByText(/predicted: 27.5/i)).toBeInTheDocument();
  expect(screen.getByText(/actual: 24/i)).toBeInTheDocument();
  expect(screen.getByText(/off by 3.5/i)).toBeInTheDocument();
});

it("shows only the predicted value for an unsettled player prop", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(detail);
  vi.mocked(api.getGamePlayers).mockResolvedValue([
    { player_id: "203999", player_name: "Nikola Jokic", stat: "points", predicted_value: 27.5, actual_value: null },
  ]);

  render(<GameDetailModal gameId="g1" onClose={() => {}} />);

  expect(await screen.findByText("points")).toBeInTheDocument();
  expect(screen.getByText("27.5")).toBeInTheDocument();
  expect(screen.queryByText(/actual/i)).not.toBeInTheDocument();
});

it("names the favoured side for the margin when both models agree", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue({ ...detail, prediction: { home_win_probability: 0.47, predicted_margin: -4.8, predicted_total: 221.3 } });
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);
  render(<GameDetailModal gameId="g1" onClose={() => {}} />);
  expect(await screen.findByText("MIA to win")).toBeInTheDocument();
  expect(screen.getByText("MIA by 4.8")).toBeInTheDocument();
});

it("calls the margin a toss-up when it disagrees with the win pick or is under half a point", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue({ ...detail, prediction: { home_win_probability: 0.52, predicted_margin: -0.6, predicted_total: 221.3 } });
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);
  const { unmount } = render(<GameDetailModal gameId="g1" onClose={() => {}} />);
  expect(await screen.findByText("BOS to win")).toBeInTheDocument();
  expect(screen.getByText("Toss-up")).toBeInTheDocument();
  expect(screen.queryByText(/MIA by/)).not.toBeInTheDocument();
  unmount();

  vi.mocked(api.getGameDetail).mockResolvedValue({ ...detail, prediction: { home_win_probability: 0.6, predicted_margin: 0.03, predicted_total: 221.3 } });
  render(<GameDetailModal gameId="g1" onClose={() => {}} />);
  expect(await screen.findByText("Toss-up")).toBeInTheDocument();
  expect(screen.queryByText(/by 0\.0/)).not.toBeInTheDocument();
});

it("labels a pick rebuilt after tip-off and never judges it", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue({ ...completedDetail, rebuilt: true });
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  expect(await screen.findByText(/Rebuilt after tip-off: BOS · 62%/)).toBeInTheDocument();
  expect(screen.getByText(/not counted/i)).toBeInTheDocument();
  expect(screen.queryByText("Called it ✓")).not.toBeInTheDocument();
  expect(screen.queryByTestId("post-match-verdict")).not.toBeInTheDocument();
});

it("says so when no pick was made before a final", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue({ ...completedDetail, prediction: null });
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  expect(await screen.findByText("No pick was made before tip-off.")).toBeInTheDocument();
});

it("names the teams and writes dates in words", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(detail);
  vi.mocked(api.getGamePlayers).mockResolvedValue(players);

  render(<GameDetailModal gameId="g1" onClose={() => {}} />);

  expect(await screen.findByRole("heading", { name: "Heat at Celtics" })).toBeInTheDocument();
  expect(screen.getByText("Thu 1 Jan")).toBeInTheDocument();
  expect(screen.queryByText("2026-01-01")).not.toBeInTheDocument();
});

it("is a labelled dialog, and a failed load offers Try again", async () => {
  vi.mocked(api.getGameDetail).mockRejectedValueOnce(new Error("x")).mockResolvedValue(detail);
  vi.mocked(api.getGamePlayers).mockResolvedValue(players);

  render(<GameDetailModal gameId="g1" onClose={() => {}} />);

  expect(await screen.findByRole("alert")).toHaveTextContent("We couldn't load this game.");
  await userEvent.click(screen.getByRole("button", { name: "Try again" }));
  expect(await screen.findByRole("dialog", { name: "Heat at Celtics" })).toHaveAttribute("aria-modal", "true");
});

it("labels rebuilt player props and market rows, and never judges them", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue({
    ...completedDetail,
    markets: [{ market: "h2h", selection: "BOS", model_probability: 0.6, market_probability: 0.5, edge: 0.1, bookmaker: "DraftKings", american_odds: -120, point: null, rebuilt: true }],
  });
  vi.mocked(api.getGamePlayers).mockResolvedValue([
    { player_id: "p1", player_name: "Jayson Tatum", stat: "points", predicted_value: 27.456, actual_value: 31, rebuilt: true },
  ]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  const row = await screen.findByTestId("market-row");
  expect(row).toHaveTextContent("Rebuilt");
  expect(row).not.toHaveTextContent("✓");
  const prop = screen.getByText("Jayson Tatum").closest("li")!;
  expect(prop).toHaveTextContent("Predicted: 27.5 — Actual: 31");
  expect(prop).not.toHaveTextContent(/off by/);
  expect(screen.getByText(/built after tip-off/i)).toBeInTheDocument();
});

it("labels a rebuilt pick on a game that has not finished", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue({ ...detail, rebuilt: true });
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g1" onClose={() => {}} />);

  expect(await screen.findByText(/Rebuilt after tip-off: BOS · 62%/)).toBeInTheDocument();
});
