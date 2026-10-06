import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import StandingsPanel from "./StandingsPanel";
import { api } from "../api/client";

vi.mock("../api/client", () => ({ api: { getHubStandings: vi.fn() } }));
afterEach(() => vi.restoreAllMocks());

describe("StandingsPanel", () => {
  it("splits standings into East and West columns by seed", async () => {
    vi.mocked(api.getHubStandings).mockResolvedValue([
      { conference: "East", seed: 1, abbreviation: "BOS", wins: 50, losses: 20, win_pct: 0.714, games_back: 0, playoff_status: "clinched", season_state: "regular" },
      { conference: "West", seed: 1, abbreviation: "LAL", wins: 48, losses: 22, win_pct: 0.686, games_back: 0, playoff_status: "clinched", season_state: "regular" },
    ]);

    render(<StandingsPanel />);

    await waitFor(() => expect(screen.getByText("BOS")).toBeInTheDocument());
    expect(screen.getByText("East")).toBeInTheDocument();
    expect(screen.getByText("West")).toBeInTheDocument();
    // Status in words, and a leader is not "0.0" games back.
    expect(screen.getAllByText("Clinched")).toHaveLength(2);
    expect(screen.queryByText("clinched")).not.toBeInTheDocument();
    expect(screen.queryByText("0.0")).not.toBeInTheDocument();
  });

  it("shows an empty state when no standings are cached yet", async () => {
    vi.mocked(api.getHubStandings).mockResolvedValue([]);
    render(<StandingsPanel />);
    await waitFor(() => expect(screen.getByText(/no standings data/i)).toBeInTheDocument());
  });
});
describe("StandingsPanel off-season", () => {
  const row = (season_state: "offseason" | "preseason" | "regular") => ({
    conference: "East" as const,
    seed: 1,
    abbreviation: "BOS",
    wins: 50,
    losses: 20,
    win_pct: 0.714,
    games_back: 0,
    playoff_status: null,
    season_state,
  });

  it("claims no playoff status off-season and says why", async () => {
    vi.mocked(api.getHubStandings).mockResolvedValue([row("offseason")]);
    render(<StandingsPanel />);
    await waitFor(() => expect(screen.getByText("BOS")).toBeInTheDocument());
    // The bug: six teams "Clinched" about a league nobody had played in.
    expect(screen.queryByText("Clinched")).not.toBeInTheDocument();
    expect(screen.queryByText(/play-in/i)).not.toBeInTheDocument();
    expect(screen.getByText(/off-season/i)).toBeInTheDocument();
    expect(screen.getByText(/last season/i)).toBeInTheDocument();
    // The record is still real arithmetic and still shown.
    expect(screen.getByText("50-20")).toBeInTheDocument();
  });

  it("does not call a loaded schedule a race", async () => {
    vi.mocked(api.getHubStandings).mockResolvedValue([row("preseason")]);
    render(<StandingsPanel />);
    await waitFor(() => expect(screen.getByText(/pre-season/i)).toBeInTheDocument());
    expect(screen.queryByText("Clinched")).not.toBeInTheDocument();
  });
});
