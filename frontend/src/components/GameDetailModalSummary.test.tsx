import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import GameDetailModal from "./GameDetailModal";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: { getGameDetail: vi.fn(), getGamePlayers: vi.fn(), explain: vi.fn() },
}));

afterEach(() => vi.restoreAllMocks());

const detail = {
  game_id: "g1", game_date: "2026-11-01", home_team: "BOS", away_team: "MIA",
  prediction: { home_win_probability: 0.62, predicted_margin: 3.5, predicted_total: 224.5 },
  completed: false, home_pts: null, away_pts: null,
  markets: [], head_to_head: [], home_recent_form: [], away_recent_form: [],
};

const explanation = {
  headline: "Boston are the slight favourites, but this is close to a coin flip.",
  sections: [{ market: "result", title: "Why Boston", text: "The model has them at 62% at home." }],
  source: "template" as const,
  model: "",
  generated_at: new Date().toISOString(),
  sport: "nba",
  pick_timing: "pre_kickoff" as const,
};

function primeGame() {
  vi.mocked(api.getGameDetail).mockResolvedValue(detail as never);
  vi.mocked(api.getGamePlayers).mockResolvedValue([] as never);
}

describe("GameDetailModal and the plain-English panel", () => {
  it("fetches the summary for this game and shows its headline", async () => {
    primeGame();
    vi.mocked(api.explain).mockResolvedValue(explanation);
    render(<GameDetailModal gameId="g1" onClose={() => {}} />);
    expect(await screen.findByText(explanation.headline)).toBeInTheDocument();
    expect(api.explain).toHaveBeenCalledWith("nba", "g1");
  });

  it("still shows the game while the summary is being written", async () => {
    primeGame();
    vi.mocked(api.explain).mockReturnValue(new Promise(() => {}));
    render(<GameDetailModal gameId="g1" onClose={() => {}} />);
    // The panel shows a labelled Skeleton, not a blank box. The game's own
    // Skeleton is on screen at this moment too, so match on the text.
    expect(screen.getAllByRole("status").some((el) => el.textContent?.includes("Writing the summary…"))).toBe(true);
    // The game's own data must not wait on the summary.
    expect(await screen.findByRole("heading", { name: "Heat at Celtics" })).toBeInTheDocument();
  });

  it("offers a retry that asks the explainer again", async () => {
    primeGame();
    vi.mocked(api.explain).mockRejectedValue(new Error("down"));
    render(<GameDetailModal gameId="g1" onClose={() => {}} />);
    const retry = await screen.findByRole("button", { name: "Try again" });
    vi.mocked(api.explain).mockResolvedValue(explanation);
    await userEvent.click(retry);
    expect(await screen.findByText(explanation.headline)).toBeInTheDocument();
    expect(api.explain).toHaveBeenCalledTimes(2);
  });

  it("leaves the game fully usable when the explainer is not deployed", async () => {
    primeGame();
    // A client with no explain method at all is the pre-Task-13 shape, and
    // the 404 case behaves the same way from the panel's point of view.
    vi.mocked(api.explain).mockRejectedValue(new Error("404"));
    render(<GameDetailModal gameId="g1" onClose={() => {}} />);
    expect(await screen.findByRole("heading", { name: "Heat at Celtics" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });
});
