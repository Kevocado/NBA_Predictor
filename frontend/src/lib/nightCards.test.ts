import { describe, expect, it } from "vitest";
import type { Game } from "../api/client";
import { dayHeading, nextUpIds, toCardModel, weekLabel, weekTally } from "./nightCards";

const TZ = "America/Chicago";
// Tue 21 Oct 2025, 7:30 PM EDT = 23:30 UTC.
const TIP = "2025-10-21T23:30Z";
const TIP_MS = Date.parse(TIP);

function game(over: Partial<Game> = {}): Game {
  return {
    game_id: "g1", game_date: "2025-10-21", tip_off: TIP, home_team: "BOS", away_team: "MIA",
    prediction: { home_win_probability: 0.62, predicted_margin: 3.5, predicted_total: 224.5 },
    rebuilt: false, completed: false, home_pts: null, away_pts: null,
    ...over,
  };
}

describe("toCardModel", () => {
  it("puts away on the left and home on the right, with team colours and nicknames", () => {
    const card = toCardModel(game(), false, TZ, TIP_MS - 3600_000);
    expect(card.left).toMatchObject({ code: "MIA", name: "Heat", color: "#98002E" });
    expect(card.right).toMatchObject({ code: "BOS", name: "Celtics", color: "#007A33" });
  });

  it("before tip-off: local tip time in the middle, the day on top, the favourite as the pick", () => {
    const card = toCardModel(game(), false, TZ, TIP_MS - 3600_000);
    expect(card.centre).toBe("6:30 PM");
    expect(card.when).toBe("Tue 21 Oct");
    expect(card.pick).toEqual({ label: "BOS", prob: 0.62 });
    expect(card.bar!.map((s) => s.label)).toEqual(["MIA", "BOS"]);
    expect(card.meta).toBe("BOS by 3.5 · Total 224.5");
    expect(card.status).toBeUndefined();
  });

  it("leads with the away team when it is favoured", () => {
    const card = toCardModel(game({ prediction: { home_win_probability: 0.47, predicted_margin: -4.8, predicted_total: 220 } }), false, TZ, 0);
    expect(card.pick).toEqual({ label: "MIA", prob: 0.53 });
    expect(card.meta).toBe("MIA by 4.8 · Total 220.0");
  });

  it("never prints a margin that contradicts the pick", () => {
    const card = toCardModel(game({ prediction: { home_win_probability: 0.55, predicted_margin: -0.6, predicted_total: 220 } }), false, TZ, 0);
    expect(card.meta).toBe("Toss-up · Total 220.0");
  });

  it("marks the next game up only when asked", () => {
    expect(toCardModel(game(), true, TZ, TIP_MS - 60_000).status).toBe("next");
  });

  it("shows Live after tip-off, then 'Awaiting result' once the game should be over", () => {
    expect(toCardModel(game(), true, TZ, TIP_MS + 3600_000).status).toBe("live");
    const late = toCardModel(game(), true, TZ, TIP_MS + 4 * 3600_000);
    expect(late.status).toBeUndefined();
    expect(late.when).toBe("Tue 21 Oct · Awaiting result");
  });

  it("judges a final on its pre-tip pick: Called it or Missed", () => {
    const won = toCardModel(game({ completed: true, home_pts: 110, away_pts: 102 }), false, TZ, TIP_MS + 86_400_000);
    expect(won.centre).toBe("102–110");
    expect(won.when).toBe("Tue 21 Oct · Final");
    expect(won.status).toBe("called");
    const lost = toCardModel(game({ completed: true, home_pts: 99, away_pts: 102 }), false, TZ, TIP_MS + 86_400_000);
    expect(lost.status).toBe("missed");
  });

  it("labels a pick rebuilt after tip-off and never judges it", () => {
    const card = toCardModel(game({ completed: true, home_pts: 110, away_pts: 102, rebuilt: true }), false, TZ, TIP_MS + 86_400_000);
    expect(card.status).toBe("rebuilt");
  });

  it("says there was no pick on a final without one", () => {
    const card = toCardModel(game({ completed: true, home_pts: 110, away_pts: 102, prediction: null }), false, TZ, TIP_MS + 86_400_000);
    expect(card.status).toBe("nopick");
    expect(card.pick).toBeUndefined();
  });

  it("without a known tip time shows the date only, and never a made-up time", () => {
    const card = toCardModel(game({ tip_off: null }), false, TZ, Date.parse("2025-10-20T12:00Z"));
    expect(card.centre).toBe("at");
    expect(card.when).toBe("Tue 21 Oct");
  });

  it("drops team colours from the bar when both teams share one", () => {
    const card = toCardModel(game({ home_team: "NOP", away_team: "MIN" }), false, TZ, 0);
    expect(card.bar!.every((s) => s.color === undefined)).toBe(true);
    expect(toCardModel(game(), false, TZ, 0).bar!.every((s) => s.color)).toBe(true);
  });
});

describe("nextUpIds", () => {
  const early = game({ game_id: "a", tip_off: "2025-10-21T23:00Z" });
  const same = game({ game_id: "b", tip_off: "2025-10-21T23:00Z" });
  const later = game({ game_id: "c", tip_off: "2025-10-22T02:00Z" });

  it("is every game at the earliest tip still to come, in the current week only", () => {
    const now = Date.parse("2025-10-21T12:00Z");
    expect([...nextUpIds([later, early, same], true, now)].sort()).toEqual(["a", "b"]);
    expect(nextUpIds([later, early], false, now).size).toBe(0);
  });

  it("ignores games without a known tip time", () => {
    expect(nextUpIds([game({ tip_off: null })], true, 0).size).toBe(0);
  });
});

describe("weekTally", () => {
  it("counts only pre-tip picks on finals, rebuilt ones apart", () => {
    const games = [
      game({ game_id: "1", completed: true, home_pts: 110, away_pts: 100 }),
      game({ game_id: "2", completed: true, home_pts: 90, away_pts: 100 }),
      game({ game_id: "3", completed: true, home_pts: 110, away_pts: 100, rebuilt: true }),
      game({ game_id: "4", completed: true, home_pts: 110, away_pts: 100, prediction: null }),
      game({ game_id: "5" }),
    ];
    expect(weekTally(games)).toEqual({ hits: 1, settled: 2, rebuilt: 1 });
  });
});

describe("labels", () => {
  it("writes the week and each day in the family's date style", () => {
    expect(weekLabel("2025-10-20")).toBe("20–26 Oct");
    expect(weekLabel("2025-10-27")).toBe("27 Oct – 2 Nov");
    expect(dayHeading("2025-10-21")).toBe("Tuesday 21 Oct");
  });
});
