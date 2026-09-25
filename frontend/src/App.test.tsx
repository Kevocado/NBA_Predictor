import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import App from "./App";

afterEach(() => {
  vi.restoreAllMocks();
});

function mockFetchEmpty() {
  vi.spyOn(globalThis, "fetch").mockResolvedValue({
    ok: true,
    json: async () => [],
  } as Response);
}

describe("App", () => {
  it("renders the site title", () => {
    mockFetchEmpty();

    render(
      <MemoryRouter>
        <App />
      </MemoryRouter>
    );
    expect(screen.getByRole("heading", { name: "NBA Predictor" })).toBeInTheDocument();
  });
});

describe("App routing", () => {
  it("navigates to the Data Hub page when its nav link is clicked", async () => {
    mockFetchEmpty();

    render(
      <MemoryRouter>
        <App />
      </MemoryRouter>
    );

    await userEvent.click(screen.getByRole("button", { name: "Data Hub" }));
    expect(await screen.findByTestId("hub-tab-team-hub")).toBeInTheDocument();
  });
});
describe("App family frame", () => {
  function renderAt(path: string) {
    mockFetchEmpty();
    return render(
      <MemoryRouter initialEntries={[path]}>
        <App />
      </MemoryRouter>
    );
  }

  it("carries the family wordmark, the NBA accent and a switcher to every sport", () => {
    const { container } = renderAt("/");
    expect(screen.getByText("Predictor")).toBeInTheDocument();
    expect(container.querySelector("[data-sport='nba']")).not.toBeNull();
    const sports = screen.getByRole("navigation", { name: "Sports" });
    expect(within(sports).getByRole("link", { name: "NBA" })).toHaveAttribute("aria-current", "page");
    expect(within(sports).getByRole("link", { name: "NFL" }).getAttribute("href")).toMatch(/sports\..*\?sport=nfl$/);
    expect(within(sports).getByRole("link", { name: "PL" })).toBeInTheDocument();
  });

  it("marks the page tab that matches the route, including on a deep link", () => {
    renderAt("/calibration-report");
    const pages = screen.getByRole("navigation", { name: "Pages" });
    expect(within(pages).getByRole("button", { name: "Calibration" })).toHaveAttribute("aria-current", "page");
    expect(within(pages).getByRole("button", { name: "Games" })).not.toHaveAttribute("aria-current");
    expect(within(pages).getAllByRole("button").map((b) => b.textContent)).toEqual(["Games", "Data Hub", "Calibration", "Model"]);
  });

  it("navigates when a page tab is pressed", async () => {
    renderAt("/");
    await userEvent.click(screen.getByRole("button", { name: "Calibration" }));
    expect(screen.getByRole("button", { name: "Calibration" })).toHaveAttribute("aria-current", "page");
  });
});
