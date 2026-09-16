import { describe, expect, it, vi, afterEach } from "vitest";
import { api } from "./client";

function mockFetchOnce(body: unknown, ok = true, status = 200) {
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok,
    status,
    json: async () => body,
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("api client", () => {
  it("getTeams fetches /teams and returns parsed JSON", async () => {
    mockFetchOnce([{ abbreviation: "BOS", name: "Boston Celtics", conference: "East", division: "Atlantic" }]);

    const teams = await api.getTeams();

    expect(fetch).toHaveBeenCalledWith(expect.stringContaining("/teams"));
    expect(teams[0].abbreviation).toBe("BOS");
  });

  it("getGames includes the date query parameter", async () => {
    mockFetchOnce([]);

    await api.getGames("2026-11-01");

    expect(fetch).toHaveBeenCalledWith(expect.stringContaining("/games?date=2026-11-01"));
  });

  it("getGameDetail fetches the game-specific path", async () => {
    mockFetchOnce({
      game_id: "g1", game_date: "2026-11-01", home_team: "BOS", away_team: "MIA",
      prediction: null, markets: [],
    });

    const detail = await api.getGameDetail("g1");

    expect(fetch).toHaveBeenCalledWith(expect.stringContaining("/games/g1"));
    expect(detail.game_id).toBe("g1");
  });

  it("throws a descriptive error when the response is not ok", async () => {
    mockFetchOnce({ detail: "not found" }, false, 404);

    await expect(api.getGameDetail("missing")).rejects.toThrow(/404/);
  });

  it("getTrackRecord fetches /hub/track-record", async () => {
    mockFetchOnce([{ market: "h2h", total_predictions: 10, correct_predictions: 6, hit_rate: 0.6 }]);

    const records = await api.getTrackRecord();

    expect(fetch).toHaveBeenCalledWith(expect.stringContaining("/hub/track-record"));
    expect(records[0].hit_rate).toBe(0.6);
  });
});