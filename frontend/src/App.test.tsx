import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
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

    await userEvent.click(screen.getByRole("link", { name: "Data Hub" }));
    expect(await screen.findByTestId("hub-tab-team-hub")).toBeInTheDocument();
  });
});