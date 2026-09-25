import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const css = readFileSync(resolve(__dirname, "index.css"), "utf8");
const html = readFileSync(resolve(__dirname, "../index.html"), "utf8");

describe("index.css", () => {
  it("loads the family fonts first and the tokens last", () => {
    const order = ["./predictor-ui/fonts.css", "tailwindcss", "./predictor-ui/tokens.css"].map((s) => css.indexOf(`@import "${s}"`));
    expect(order.every((i) => i >= 0)).toBe(true);
    expect(order).toEqual([...order].sort((a, b) => a - b));
  });

  it("drops the old faces, in the CSS and the page head", () => {
    for (const face of [/Big Shoulders/, /Manrope/]) {
      expect(css).not.toMatch(face);
      expect(html).not.toMatch(face);
    }
  });

  it("maps every legacy court colour onto a family token", () => {
    const legacy = [...css.matchAll(/--color-(?:court|line|hardwood|net|win|shotclock)[\w-]*:\s*([^;]+);/g)].map((m) => m[1].trim());
    expect(legacy.length).toBeGreaterThanOrEqual(9);
    for (const value of legacy) expect(value).toMatch(/^var\(--color-pr-/);
  });

  it("re-resolves the aliases under data-sport, so the accent is NBA orange, not the Hub's white", () => {
    expect(css).toMatch(/:root,\s*\[data-sport\]\s*\{[^}]*--color-hardwood:\s*var\(--color-pr-accent\)/);
  });
});

describe("tables on phones", () => {
  it("keep a gutter between cells", () => {
    expect(css).toMatch(/td\s*\{[^}]*padding:\s*0\.375rem 0\.75rem/);
  });
});
