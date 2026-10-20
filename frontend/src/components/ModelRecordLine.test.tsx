import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ModelRecordLine, preTipRecord } from "./ModelRecordLine";
import type { TrackRecord } from "../api/client";

const row = (over: Partial<TrackRecord>): TrackRecord => ({
  market: "game_outcome",
  total_predictions: 0,
  correct_predictions: 0,
  hit_rate: null,
  settled: true,
  ...over,
});

const gameOutcome = row({
  total_predictions: 50,
  correct_predictions: 40,
  pre_tip: { total_predictions: 66, correct_predictions: 33, hit_rate: 0.5 },
});
const h2h = row({
  market: "h2h",
  total_predictions: 20,
  correct_predictions: 19,
  pre_tip: { total_predictions: 20, correct_predictions: 19, hit_rate: 0.95 },
});

describe("preTipRecord", () => {
  it("reads the game_outcome pre-tip block, not the headline or another market", () => {
    expect(preTipRecord([h2h, gameOutcome])).toEqual({ hits: 33, settled: 66 });
  });

  it("is silent without a game_outcome row", () => {
    expect(preTipRecord([h2h])).toBeNull();
  });

  it("is silent when the pre-tip block is missing or empty", () => {
    expect(preTipRecord([row({ total_predictions: 50, correct_predictions: 40 })])).toBeNull();
    expect(
      preTipRecord([row({ pre_tip: { total_predictions: 0, correct_predictions: 0, hit_rate: null } })]),
    ).toBeNull();
  });

  it("is silent when the feed fails", () => {
    expect(preTipRecord(null)).toBeNull();
  });
});

describe("ModelRecordLine", () => {
  it("says the count and links the record", () => {
    render(<ModelRecordLine rows={[gameOutcome]} />);
    expect(screen.getByText(/Model, picks made before tip-off:/)).toHaveTextContent(
      "Model, picks made before tip-off: 33 of 66",
    );
    expect(screen.getByRole("link", { name: "Track record" })).toHaveAttribute("href", "/track-record");
  });

  it("can state the same record as a percentage", () => {
    render(<ModelRecordLine rows={[gameOutcome]} format="percent" />);
    expect(screen.getByText(/Model, picks made before tip-off:/)).toHaveTextContent(
      "Model, picks made before tip-off: 50%",
    );
  });

  it("renders nothing at all when the record is unavailable", () => {
    const { container } = render(<ModelRecordLine rows={null} />);
    expect(container).toBeEmptyDOMElement();
  });
});
