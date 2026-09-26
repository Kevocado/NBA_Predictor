import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { GameSummaryPanel } from "./GameSummaryPanel";

const explanation = {
  headline: "Boston are the slight favourites, but this is close to a coin flip.",
  sections: [{ market: "result", title: "Why Boston", text: "The model has them at 62% at home." }],
  source: "template" as const,
  model: "",
  generated_at: new Date().toISOString(),
  sport: "nba",
  pick_timing: "pre_kickoff" as const,
};

describe("GameSummaryPanel", () => {
  it("asks for this game's summary and shows its headline", async () => {
    const fetcher = vi.fn().mockResolvedValue(explanation);
    render(<GameSummaryPanel gameId="g1" fetcher={fetcher} />);
    expect(await screen.findByText(explanation.headline)).toBeInTheDocument();
    expect(fetcher).toHaveBeenCalledWith("nba", "g1");
  });

  it("says it is writing, which is a state and not a blank space", () => {
    const fetcher = vi.fn().mockReturnValue(new Promise<typeof explanation>(() => {}));
    render(<GameSummaryPanel gameId="g1" fetcher={fetcher} />);
    expect(screen.getByRole("status")).toHaveTextContent("Writing the summary…");
  });

  it("offers a retry that asks again, and recovers", async () => {
    const fetcher = vi.fn().mockRejectedValue(new Error("explainer down"));
    render(<GameSummaryPanel gameId="g1" fetcher={fetcher} />);
    const retry = await screen.findByRole("button", { name: "Try again" });
    fetcher.mockResolvedValue(explanation);
    await userEvent.click(retry);
    expect(await screen.findByText(explanation.headline)).toBeInTheDocument();
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("renders nothing at all when there is no explainer", () => {
    const { container } = render(<GameSummaryPanel gameId="g1" />);
    expect(container).toBeEmptyDOMElement();
  });

  it("uses the NBA's moment in the rebuilt status", async () => {
    const fetcher = vi.fn().mockResolvedValue({ ...explanation, pick_timing: "rebuilt" });
    render(<GameSummaryPanel gameId="g1" fetcher={fetcher} />);
    // Basketball says tip-off, not kickoff: the same word the site's own
    // StatusBadge uses everywhere else.
    expect(await screen.findByText("Rebuilt after tip-off")).toBeInTheDocument();
    expect(screen.getByText(/not counted/)).toBeInTheDocument();
  });
});
