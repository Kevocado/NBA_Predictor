import { afterEach, describe, expect, it } from "vitest";
import { addDays, mondayOf } from "./weeks";

const original = process.env.TZ;
afterEach(() => {
  process.env.TZ = original;
});

describe("week arithmetic", () => {
  it.each(["America/Chicago", "Africa/Nairobi", "Asia/Tokyo", "Pacific/Honolulu"])("steps whole weeks, Monday to Monday, in %s", (tz) => {
    process.env.TZ = tz;
    expect(addDays("2026-02-16", 7)).toBe("2026-02-23");
    expect(addDays("2026-02-16", -7)).toBe("2026-02-09");
    expect(addDays("2026-03-02", 7)).toBe("2026-03-09"); // across the US clock change
  });

  it("finds the Monday of the viewer's own calendar day", () => {
    process.env.TZ = "Asia/Tokyo";
    // Sunday 22 Feb 23:30 in Tokyo is still Sunday there: its Monday is the 16th.
    expect(mondayOf(new Date("2026-02-22T14:30:00Z"))).toBe("2026-02-16");
    process.env.TZ = "Pacific/Honolulu";
    expect(mondayOf(new Date("2026-02-23T08:00:00Z"))).toBe("2026-02-16");
  });
});
