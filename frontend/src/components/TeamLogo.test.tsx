import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import TeamLogo from "./TeamLogo";

describe("TeamLogo", () => {
  it("renders the team logo image with an accessible label", () => {
    render(<TeamLogo team="BOS" />);
    const img = screen.getByRole("img", { name: "BOS logo" });
    expect(img.tagName).toBe("IMG");
    expect(img).toHaveAttribute("src", expect.stringContaining("bos.png"));
  });

  it("falls back to the first uppercase character for unknown teams", () => {
    render(<TeamLogo team="xyz" />);
    expect(screen.getByText("X")).toBeInTheDocument();
  });

  it("falls back to the team initial when the image fails to load", () => {
    render(<TeamLogo team="BOS" />);
    fireEvent.error(screen.getByRole("img", { name: "BOS logo" }));
    expect(screen.getByText("B")).toBeInTheDocument();
  });
});
