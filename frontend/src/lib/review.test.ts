import { describe, expect, it } from "vitest";
import { reviewPlayers, type ReviewActual, type ReviewProjection } from "./review";

const BANDS = { points: 4.6, rebounds: 1.9, assists: 1.3, threes: 0.9 };

const projection = (
  player_id: string,
  stat: string,
  value: number,
  over: Partial<ReviewProjection> = {},
): ReviewProjection => ({
  player_id,
  player_name: `Player ${player_id}`,
  stat,
  value,
  ...over,
});

const actual = (player_id: string, stat: string, value: number): ReviewActual => ({
  player_id,
  stat,
  value,
});

describe("reviewPlayers", () => {
  it("beats the projection only when the actual is more than the band above it", () => {
    const result = reviewPlayers(
      [projection("p1", "points", 20)],
      [actual("p1", "points", 25)],
      BANDS,
    );

    expect(result.beat).toHaveLength(1);
    expect(result.beat[0].gap).toBeCloseTo(5);
    expect(result.short).toHaveLength(0);
    expect(result.onNumber).toBe(0);
    expect(result.total).toBe(1);
  });

  it("calls a gap of exactly the band on the number, not a beating", () => {
    // 4.6 above the projection is the model's own error, exactly.
    const result = reviewPlayers(
      [projection("p1", "points", 20)],
      [actual("p1", "points", 24.6)],
      BANDS,
    );

    expect(result.beat).toHaveLength(0);
    expect(result.short).toHaveLength(0);
    expect(result.onNumber).toBe(1);
    expect(result.total).toBe(1);
  });

  it("calls a miss of exactly the band on the number too", () => {
    const result = reviewPlayers(
      [projection("p1", "points", 20)],
      [actual("p1", "points", 15.4)],
      BANDS,
    );

    expect(result.short).toHaveLength(0);
    expect(result.onNumber).toBe(1);
  });

  it("says a player fell short when the actual is more than the band below it", () => {
    const result = reviewPlayers(
      [projection("p1", "points", 20)],
      [actual("p1", "points", 14)],
      BANDS,
    );

    expect(result.short).toHaveLength(1);
    expect(result.short[0].gap).toBeCloseTo(-6);
    expect(result.beat).toHaveLength(0);
  });

  it("leaves out a player who did not play", () => {
    const result = reviewPlayers(
      [projection("p1", "points", 20), projection("p2", "points", 30)],
      [actual("p1", "points", 40)],
      BANDS,
    );

    expect(result.beat).toHaveLength(1);
    expect(result.beat[0].player_id).toBe("p1");
    expect(result.total).toBe(1);
  });

  it("never judges a projection rebuilt after tip-off", () => {
    const result = reviewPlayers(
      [projection("p1", "points", 20, { rebuilt: true })],
      [actual("p1", "points", 40)],
      BANDS,
    );

    expect(result.beat).toHaveLength(0);
    expect(result.total).toBe(0);
  });

  it("judges nothing without bands, so the section has nothing to say", () => {
    const result = reviewPlayers(
      [projection("p1", "points", 20)],
      [actual("p1", "points", 40)],
      null,
    );

    expect(result.beat).toHaveLength(0);
    expect(result.short).toHaveLength(0);
    expect(result.onNumber).toBe(0);
    expect(result.total).toBe(0);
  });

  it("judges nothing for a stat with no band of its own", () => {
    const result = reviewPlayers(
      [projection("p1", "steals", 3)],
      [actual("p1", "steals", 9)],
      BANDS,
    );

    expect(result.total).toBe(0);
  });

  it("keeps the five biggest gaps in each column", () => {
    const projections = Array.from({ length: 8 }, (_, i) => projection(`p${i}`, "points", 20));
    const actuals = Array.from({ length: 8 }, (_, i) => actual(`p${i}`, "points", 30 + i));

    const result = reviewPlayers(projections, actuals, BANDS);

    expect(result.beat).toHaveLength(5);
    // The biggest gap first: the player who scored 37 against a projection of 20.
    expect(result.beat[0].player_id).toBe("p7");
    expect(result.total).toBe(8);
  });

  it("counts the lines that were on the number", () => {
    const result = reviewPlayers(
      [
        projection("p1", "points", 20),
        projection("p2", "points", 20),
        projection("p3", "points", 20),
      ],
      [actual("p1", "points", 22), actual("p2", "points", 30), actual("p3", "points", 18)],
      BANDS,
    );

    expect(result.onNumber).toBe(2);
    expect(result.total).toBe(3);
    expect(result.beat).toHaveLength(1);
  });
});
