import { describe, expect, it, vi, afterEach } from "vitest";
import { api, REQUEST_TIMEOUT_MS } from "./client";

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

    expect(fetch).toHaveBeenCalledWith(expect.stringContaining("/teams"), expect.anything());
    expect(teams[0].abbreviation).toBe("BOS");
  });

  it("getGames includes the date query parameter", async () => {
    mockFetchOnce([]);

    await api.getGames("2026-11-01");

    expect(fetch).toHaveBeenCalledWith(expect.stringContaining("/games?date=2026-11-01"), expect.anything());
  });

  it("getGameDetail fetches the game-specific path", async () => {
    mockFetchOnce({
      game_id: "g1", game_date: "2026-11-01", home_team: "BOS", away_team: "MIA",
      prediction: null, markets: [],
    });

    const detail = await api.getGameDetail("g1");

    expect(fetch).toHaveBeenCalledWith(expect.stringContaining("/games/g1"), expect.anything());
    expect(detail.game_id).toBe("g1");
  });

  it("throws a descriptive error when the response is not ok", async () => {
    mockFetchOnce({ detail: "not found" }, false, 404);

    await expect(api.getGameDetail("missing")).rejects.toThrow(/404/);
  });

  it("getTrackRecord fetches /hub/track-record", async () => {
    mockFetchOnce([{ market: "h2h", total_predictions: 10, correct_predictions: 6, hit_rate: 0.6 }]);

    const records = await api.getTrackRecord();

    expect(fetch).toHaveBeenCalledWith(expect.stringContaining("/hub/track-record"), expect.anything());
    expect(records[0].hit_rate).toBe(0.6);
  });

  it("getVsMarket fetches /hub/vs-market", async () => {
    mockFetchOnce({
      market: "h2h",
      n: 3,
      mean_model_probability: 0.57,
      mean_market_probability: 0.54,
      mean_edge_points: 3.2,
      disagreement_n: 1,
      disagreement_hit_rate: 1,
      disagreement_game_ids: ["g1"],
      weekly: [],
      scope: {
        population: "finished games with a pre-tip moneyline price",
        weekly_from: "2026-09-21",
        weekly_through: "2026-09-28",
        n_games_total: 3,
        n_games_in_weekly: 3,
        n_games_outside_weekly: 0,
      },
      method: { edge: "Edge is the model's probability for its pick minus the price." },
    });

    const vs = await api.getVsMarket();

    expect(fetch).toHaveBeenCalledWith(expect.stringContaining("/hub/vs-market"), expect.anything());
    expect(vs.n).toBe(3);
    expect(vs.scope.n_games_in_weekly + vs.scope.n_games_outside_weekly).toBe(vs.scope.n_games_total);
    expect(vs.method.edge).toContain("Edge is the model's probability");
  });
});
  it("getGamePlayers response includes actual_value", async () => {
    mockFetchOnce([{ player_id: "203999", player_name: "Nikola Jokic", stat: "points", predicted_value: 27.5, actual_value: 24.0 }]);

    const players = await api.getGamePlayers("g1");

    expect(players[0].actual_value).toBe(24.0);
  });

describe("request timeout", () => {
  it("rejects a request the server never answers, so the page can show its error state", async () => {
    vi.useFakeTimers();
    try {
      globalThis.fetch = vi.fn().mockImplementation((_u: string, init?: RequestInit) =>
        new Promise((_resolve, reject) =>
          init?.signal?.addEventListener("abort", () => reject(new DOMException("timed out", "AbortError"))),
        ));
      const assertion = expect(api.getTeams()).rejects.toThrow();
      await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS + 1);
      await assertion;
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("the explain path builds a safe route", () => {
  it("percent-encodes a game id that would otherwise split the route", async () => {
    // The explainer route is /explain/{sport}/{id}, and NBA's own proxy
    // refuses a multi-segment id with the fixed 502 rather than resolving it.
    // A game id is never one of those today, so this is insurance on a path the
    // router would happily corrupt — and the assertion is on the URL that goes
    // out, not on the argument, which is identical either way.
    mockFetchOnce({ verdict: "v", band: "moderate", factors: [] });

    await api.explainGame("401585224/../admin");

    const url = String((fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0][0]);
    expect(url).toContain(encodeURIComponent("401585224/../admin"));
    expect(url).not.toContain("401585224/../admin");
  });

  it("leaves an ordinary id untouched", async () => {
    // The control, and the one that matters most: encoding must not mangle the
    // normal case, or every request would 404 instead.
    mockFetchOnce({ verdict: "v", band: "moderate", factors: [] });

    await api.explainGame("401585224");

    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/explain/nba/401585224"),
      expect.anything(),
    );
  });
});
