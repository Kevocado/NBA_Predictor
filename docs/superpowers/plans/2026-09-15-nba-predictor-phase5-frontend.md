# NBA Predictor — Phase 5: Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the React frontend against the Phase 4 API: a Games page
with a per-game detail modal, a tabbed Data Hub (Team Hub, Player Hub, Power
Rankings, Projected Standings, Track Record), and a Model Summary page — at
the same level of detail as PL_Predictor's frontend, per spec §6.

**Architecture:** React 19 + TypeScript + Vite, styled with Tailwind CSS v4
(no component library, hand-built components, matching PL_Predictor). A
single typed `api/client.ts` module wraps every Phase 4 endpoint; every page
and component consumes it through that one module so there is exactly one
place that knows the backend's URL shape. Client-side routing via
`react-router-dom`. Tests use `vitest` + `@testing-library/react`, mocking
`api/client.ts` (never `fetch` directly) so component tests describe
behavior, not transport.

**Tech Stack:** React 19, TypeScript, Vite, Tailwind CSS v4
(`@tailwindcss/vite`), Recharts, `react-router-dom`, `vitest`,
`@testing-library/react`, `@testing-library/jest-dom`, `jsdom`.

**Spec:** [docs/superpowers/specs/2026-09-15-nba-predictor-design.md](../specs/2026-09-15-nba-predictor-design.md)

## Global Constraints

- Every network call goes through `frontend/src/api/client.ts` — no
  component calls `fetch` directly.
- Tailwind v4 theme tokens live in `frontend/src/index.css` under `:root`,
  following PL_Predictor's tonal-ramp pattern but with the NBA design
  identity from the spec: navy `--color-nba-950`…`--color-nba-500` ramp plus
  an orange accent, not PL's purple.
- Dark-mode-first (`color-scheme: dark` on `html, body`), Inter font via
  Google Fonts — matches every sibling project.
  `["East", "West"]` are the only conference strings ever rendered.
- Every page component must render a loading state, an empty state, and an
  error state distinctly (no silent blank screens) — each is exercised by a
  test.

---

## File Structure

```
frontend/
  index.html
  package.json
  tsconfig.json
  tsconfig.node.json
  vite.config.ts
  src/
    main.tsx
    App.tsx
    index.css
    api/
      client.ts
      client.test.ts
    pages/
      GamesPage.tsx
      GamesPage.test.tsx
      DataHubPage.tsx
      DataHubPage.test.tsx
      ModelSummaryPage.tsx
      ModelSummaryPage.test.tsx
    components/
      GameCard.tsx
      GameDetailModal.tsx
      GameDetailModal.test.tsx
      TeamHubPanel.tsx
      PlayerHubPanel.tsx
      PowerRankingsPanel.tsx
      StandingsPanel.tsx
      TrackRecordPanel.tsx
    test/
      setup.ts
```

---

### Task 1: Vite + React + TypeScript + Tailwind v4 scaffold with vitest

**Files:**
- Create: `frontend/package.json`, `frontend/tsconfig.json`, `frontend/tsconfig.node.json`, `frontend/vite.config.ts`, `frontend/index.html`
- Create: `frontend/src/main.tsx`, `frontend/src/App.tsx`, `frontend/src/index.css`
- Create: `frontend/src/test/setup.ts`
- Test: `frontend/src/App.test.tsx`

**Interfaces:**
- Produces: a bootable Vite dev server, a working `npm run build`, and a
  working `npm test` (vitest) — every later task's tests run through this
  setup. `App` renders a `<header>` with the site title `"NBA Predictor"`
  and a `<nav>` (routes added in Task 8).

- [ ] **Step 1: Create `frontend/package.json`**

```json
{
  "name": "nba-predictor-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "dependencies": {
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "react-router-dom": "^7.0.0",
    "recharts": "^2.13.0"
  },
  "devDependencies": {
    "@tailwindcss/vite": "^4.0.0",
    "@testing-library/jest-dom": "^6.6.0",
    "@testing-library/react": "^16.0.0",
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@vitejs/plugin-react": "^4.3.0",
    "jsdom": "^25.0.0",
    "tailwindcss": "^4.0.0",
    "typescript": "^5.6.0",
    "vite": "^6.0.0",
    "vitest": "^2.1.0"
  }
}
```

- [ ] **Step 2: Create `frontend/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "types": ["vitest/globals", "@testing-library/jest-dom"]
  },
  "include": ["src"]
}
```

- [ ] **Step 3: Create `frontend/tsconfig.node.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "skipLibCheck": true
  },
  "include": ["vite.config.ts"]
}
```

- [ ] **Step 4: Create `frontend/vite.config.ts`**

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts",
  },
});
```

- [ ] **Step 5: Create `frontend/index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>NBA Predictor</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 6: Create `frontend/src/index.css`**

```css
@import "tailwindcss";

:root {
  --color-nba-950: #050b16;
  --color-nba-900: #0a1730;
  --color-nba-850: #0f2140;
  --color-nba-800: #142b52;
  --color-nba-700: #1c3a6e;
  --color-nba-600: #26518f;
  --color-nba-border: #1e3560;
  --color-nba-orange: #ff7a1a;
  --color-nba-amber: #ffb020;
  --color-nba-text: #f2f5fa;
  --color-nba-text-dim: #a9b7cf;
  --color-nba-text-faint: #7086ab;
  --color-win: #22c55e;
  --color-loss: #ef4444;
}

html, body {
  margin: 0;
  min-height: 100vh;
  color-scheme: dark;
}

body {
  background:
    radial-gradient(circle at 15% -10%, var(--color-nba-700) 0%, transparent 45%),
    radial-gradient(circle at 100% 0%, var(--color-nba-600) 0%, transparent 35%),
    var(--color-nba-950);
  color: var(--color-nba-text);
  font-family: "Inter", ui-sans-serif, system-ui, sans-serif;
}
```

- [ ] **Step 7: Create `frontend/src/test/setup.ts`**

```ts
import "@testing-library/jest-dom/vitest";
```

- [ ] **Step 8: Write the failing test `frontend/src/App.test.tsx`**

```tsx
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import App from "./App";

describe("App", () => {
  it("renders the site title", () => {
    render(
      <MemoryRouter>
        <App />
      </MemoryRouter>
    );
    expect(screen.getByRole("heading", { name: "NBA Predictor" })).toBeInTheDocument();
  });
});
```

- [ ] **Step 9: Write `frontend/src/App.tsx`**

```tsx
export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-[var(--color-nba-border)] px-6 py-4">
        <h1 className="text-xl font-bold tracking-tight">NBA Predictor</h1>
      </header>
      <main className="px-6 py-8">
        <p className="text-[var(--color-nba-text-dim)]">
          Routes are added in Task 8.
        </p>
      </main>
    </div>
  );
}
```

- [ ] **Step 10: Write `frontend/src/main.tsx`**

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>
);
```

- [ ] **Step 11: Install dependencies and run the test**

```bash
cd "/Users/sigey/Documents/Projects/NBA_Predictor/frontend"
npm install
npm test
```

Expected: 1 passed.

- [ ] **Step 12: Verify the production build works**

```bash
npm run build
```

Expected: exits 0, produces `frontend/dist/`.

- [ ] **Step 13: Commit**

```bash
cd "/Users/sigey/Documents/Projects/NBA_Predictor"
git add frontend/
git commit -m "chore: scaffold Vite + React + TypeScript + Tailwind v4 frontend"
```

---

### Task 2: Typed API client

**Files:**
- Create: `frontend/src/api/client.ts`
- Test: `frontend/src/api/client.test.ts`

**Interfaces:**
- Consumes: nothing (wraps `fetch` against the Phase 4 API).
- Produces (all exported from `api/client.ts`):
  - Types: `Team`, `Prediction`, `Game`, `MarketPrediction`, `GameDetail`, `PlayerProp`, `TrackRecord`, `TeamHubRow`, `PlayerHubRow`, `PowerRankingRow`, `StandingsRow`.
  - `api.getTeams(): Promise<Team[]>`
  - `api.getTeam(abbreviation: string): Promise<Team>`
  - `api.getGames(date: string): Promise<Game[]>`
  - `api.getGameDetail(gameId: string): Promise<GameDetail>`
  - `api.getGamePlayers(gameId: string): Promise<PlayerProp[]>`
  - `api.getHubTeams(): Promise<TeamHubRow[]>`
  - `api.getHubPlayers(): Promise<PlayerHubRow[]>`
  - `api.getHubRankings(): Promise<PowerRankingRow[]>`
  - `api.getHubStandings(): Promise<StandingsRow[]>`
  - `api.getTrackRecord(): Promise<TrackRecord[]>`

- [ ] **Step 1: Write the failing tests**

```ts
// frontend/src/api/client.test.ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./client";

function mockFetchOnce(body: unknown, ok = true, status = 200) {
  global.fetch = vi.fn().mockResolvedValue({
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test -- api/client.test.ts`
Expected: FAIL — `Cannot find module './client'`.

- [ ] **Step 3: Write `frontend/src/api/client.ts`**

```ts
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export interface Team {
  abbreviation: string;
  name: string;
  conference: "East" | "West";
  division: string;
}

export interface Prediction {
  home_win_probability: number;
  predicted_margin: number;
  predicted_total: number;
}

export interface Game {
  game_id: string;
  game_date: string;
  home_team: string;
  away_team: string;
  prediction: Prediction | null;
}

export interface MarketPrediction {
  market: string;
  selection: string;
  model_probability: number;
  market_probability: number | null;
  edge: number | null;
  bookmaker: string | null;
  american_odds: number | null;
}

export interface GameDetail extends Game {
  markets: MarketPrediction[];
}

export interface PlayerProp {
  player_id: string;
  player_name: string;
  stat: string;
  predicted_value: number;
}

export interface TrackRecord {
  market: string;
  total_predictions: number;
  correct_predictions: number;
  hit_rate: number;
}

export interface TeamHubRow {
  abbreviation: string;
  conference: "East" | "West";
  division: string;
  wins: number;
  losses: number;
  points_per_game: number;
  opp_points_per_game: number;
  net_rating: number;
  pace: number;
  streak: number;
}

export interface PlayerHubRow {
  player_id: string;
  player_name: string;
  team: string;
  position: string;
  rating: number;
  live_form_rating: number;
  points_per_game: number;
  rebounds_per_game: number;
  assists_per_game: number;
  fg_pct: number;
  three_pt_pct: number;
  ft_pct: number;
  usage_rate: number;
  minutes_per_game: number;
}

export interface PowerRankingRow {
  rank: number;
  abbreviation: string;
  power_rating: number;
  trend: "up" | "down" | "steady";
}

export interface StandingsRow {
  conference: "East" | "West";
  seed: number;
  abbreviation: string;
  wins: number;
  losses: number;
  win_pct: number;
  games_back: number;
  playoff_status: "clinched" | "play-in" | "eliminated" | "in-hunt";
}

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`Request to ${path} failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  getTeams: () => fetchJson<Team[]>("/teams"),
  getTeam: (abbreviation: string) => fetchJson<Team>(`/teams/${abbreviation}`),
  getGames: (date: string) => fetchJson<Game[]>(`/games?date=${date}`),
  getGameDetail: (gameId: string) => fetchJson<GameDetail>(`/games/${gameId}`),
  getGamePlayers: (gameId: string) => fetchJson<PlayerProp[]>(`/games/${gameId}/players`),
  getHubTeams: () => fetchJson<TeamHubRow[]>("/hub/teams"),
  getHubPlayers: () => fetchJson<PlayerHubRow[]>("/hub/players"),
  getHubRankings: () => fetchJson<PowerRankingRow[]>("/hub/rankings"),
  getHubStandings: () => fetchJson<StandingsRow[]>("/hub/standings"),
  getTrackRecord: () => fetchJson<TrackRecord[]>("/hub/track-record"),
};
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test -- api/client.test.ts`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
cd "/Users/sigey/Documents/Projects/NBA_Predictor"
git add frontend/src/api/
git commit -m "feat: add typed API client"
```

---

### Task 3: Games page (list + `GameCard`)

**Files:**
- Create: `frontend/src/pages/GamesPage.tsx`
- Create: `frontend/src/components/GameCard.tsx`
- Test: `frontend/src/pages/GamesPage.test.tsx`

**Interfaces:**
- Consumes: `api.getGames` (Task 2).
- Produces:
  - `GameCard(props: { game: Game; onSelect: (gameId: string) => void }) -> JSX.Element` — shows matchup, date, and (if present) `prediction.home_win_probability` as a percentage; renders a "Prediction pending" badge when `prediction` is `null`.
  - `GamesPage()` — a date-picker input (defaults to today, `YYYY-MM-DD`), fetches games for the selected date on mount and on date change, renders a `GameCard` per game (empty state: `"No games scheduled for this date."`; loading state: `"Loading games…"`; error state: `"Couldn't load games."`). Clicking a card calls `setSelectedGameId`, which Task 4 will use to open the detail modal — for this task, exposed for testing via a `data-testid="game-card-{gameId}"` attribute on each card.

- [ ] **Step 1: Write the failing tests**

```tsx
// frontend/src/pages/GamesPage.test.tsx
import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import GamesPage from "./GamesPage";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: { getGames: vi.fn() },
}));

afterEach(() => {
  vi.restoreAllMocks();
});

describe("GamesPage", () => {
  it("shows a loading state before games arrive", () => {
    vi.mocked(api.getGames).mockReturnValue(new Promise(() => {}));
    render(<GamesPage />);
    expect(screen.getByText(/loading games/i)).toBeInTheDocument();
  });

  it("renders a game card per returned game", async () => {
    vi.mocked(api.getGames).mockResolvedValue([
      {
        game_id: "g1", game_date: "2026-11-01", home_team: "BOS", away_team: "MIA",
        prediction: { home_win_probability: 0.62, predicted_margin: 3.5, predicted_total: 224.5 },
      },
    ]);

    render(<GamesPage />);

    await waitFor(() => expect(screen.getByTestId("game-card-g1")).toBeInTheDocument());
    expect(screen.getByText(/62%/)).toBeInTheDocument();
  });

  it("shows an empty state when no games are scheduled", async () => {
    vi.mocked(api.getGames).mockResolvedValue([]);
    render(<GamesPage />);

    await waitFor(() => expect(screen.getByText(/no games scheduled/i)).toBeInTheDocument());
  });

  it("shows an error state when the fetch fails", async () => {
    vi.mocked(api.getGames).mockRejectedValue(new Error("network error"));
    render(<GamesPage />);

    await waitFor(() => expect(screen.getByText(/couldn't load games/i)).toBeInTheDocument());
  });

  it("re-fetches games when the date input changes", async () => {
    vi.mocked(api.getGames).mockResolvedValue([]);
    render(<GamesPage />);

    await waitFor(() => expect(api.getGames).toHaveBeenCalledTimes(1));

    const dateInput = screen.getByLabelText(/date/i);
    await userEvent.clear(dateInput);
    await userEvent.type(dateInput, "2026-12-25");

    await waitFor(() => expect(api.getGames).toHaveBeenLastCalledWith("2026-12-25"));
  });
});
```

Add `"@testing-library/user-event": "^14.5.0"` to `frontend/package.json`'s
`devDependencies` and re-run `npm install` before this test can pass.

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test -- pages/GamesPage.test.tsx`
Expected: FAIL — `Cannot find module './GamesPage'`.

- [ ] **Step 3: Write `frontend/src/components/GameCard.tsx`**

```tsx
import type { Game } from "../api/client";

interface GameCardProps {
  game: Game;
  onSelect: (gameId: string) => void;
}

export default function GameCard({ game, onSelect }: GameCardProps) {
  return (
    <button
      data-testid={`game-card-${game.game_id}`}
      onClick={() => onSelect(game.game_id)}
      className="w-full rounded-xl border border-[var(--color-nba-border)] bg-[var(--color-nba-900)]/60 p-4 text-left transition hover:border-[var(--color-nba-orange)]"
    >
      <div className="text-sm text-[var(--color-nba-text-faint)]">{game.game_date}</div>
      <div className="mt-1 font-semibold">
        {game.away_team} @ {game.home_team}
      </div>
      {game.prediction ? (
        <div className="mt-2 text-sm text-[var(--color-nba-amber)]">
          {game.home_team} win prob: {Math.round(game.prediction.home_win_probability * 100)}%
        </div>
      ) : (
        <div className="mt-2 text-sm text-[var(--color-nba-text-dim)]">Prediction pending</div>
      )}
    </button>
  );
}
```

- [ ] **Step 4: Write `frontend/src/pages/GamesPage.tsx`**

```tsx
import { useEffect, useState } from "react";
import GameCard from "../components/GameCard";
import { api, type Game } from "../api/client";

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

export default function GamesPage() {
  const [date, setDate] = useState(today());
  const [games, setGames] = useState<Game[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null);

  useEffect(() => {
    setGames(null);
    setError(null);
    api
      .getGames(date)
      .then(setGames)
      .catch(() => setError("Couldn't load games."));
  }, [date]);

  return (
    <div>
      <label className="mb-4 flex items-center gap-2 text-sm text-[var(--color-nba-text-dim)]">
        Date
        <input
          type="date"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          className="rounded border border-[var(--color-nba-border)] bg-[var(--color-nba-900)] px-2 py-1"
        />
      </label>

      {error && <p className="text-[var(--color-loss)]">{error}</p>}
      {!error && games === null && <p>Loading games…</p>}
      {!error && games !== null && games.length === 0 && <p>No games scheduled for this date.</p>}

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {games?.map((game) => (
          <GameCard key={game.game_id} game={game} onSelect={setSelectedGameId} />
        ))}
      </div>

      {selectedGameId && <p className="mt-4 text-sm">Selected game: {selectedGameId} (modal wired in Task 4)</p>}
    </div>
  );
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npm test -- pages/GamesPage.test.tsx`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
cd "/Users/sigey/Documents/Projects/NBA_Predictor"
git add frontend/package.json frontend/src/pages/GamesPage.tsx frontend/src/pages/GamesPage.test.tsx frontend/src/components/GameCard.tsx
git commit -m "feat: add Games page with date picker and game cards"
```

---

### Task 4: Game detail modal

**Files:**
- Create: `frontend/src/components/GameDetailModal.tsx`
- Test: `frontend/src/components/GameDetailModal.test.tsx`
- Modify: `frontend/src/pages/GamesPage.tsx`

**Interfaces:**
- Consumes: `api.getGameDetail`, `api.getGamePlayers` (Task 2).
- Produces:
  - `GameDetailModal(props: { gameId: string; onClose: () => void }) -> JSX.Element` — fetches detail + player props on mount, shows loading/error states, renders: matchup header, win-probability/margin/total summary, a markets table (market, selection, model % vs market %, edge %, bookmaker) sorted by `edge` descending, and a player-props list (player, stat, predicted value). Closing via an "×" button or clicking the backdrop calls `onClose`.
  - `GamesPage` is modified to render `<GameDetailModal gameId={selectedGameId} onClose={() => setSelectedGameId(null)} />` when `selectedGameId` is set.

- [ ] **Step 1: Write the failing tests**

```tsx
// frontend/src/components/GameDetailModal.test.tsx
import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import GameDetailModal from "./GameDetailModal";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: { getGameDetail: vi.fn(), getGamePlayers: vi.fn() },
}));

afterEach(() => {
  vi.restoreAllMocks();
});

const detail = {
  game_id: "g1", game_date: "2026-11-01", home_team: "BOS", away_team: "MIA",
  prediction: { home_win_probability: 0.62, predicted_margin: 3.5, predicted_total: 224.5 },
  markets: [
    { market: "h2h", selection: "home", model_probability: 0.62, market_probability: 0.55, edge: 0.07, bookmaker: "DraftKings", american_odds: -130 },
    { market: "spread", selection: "home", model_probability: 0.52, market_probability: 0.5, edge: 0.02, bookmaker: "DraftKings", american_odds: -110 },
  ],
};

const players = [{ player_id: "203999", player_name: "203999", stat: "points", predicted_value: 27.5 }];

describe("GameDetailModal", () => {
  it("shows a loading state before data arrives", () => {
    vi.mocked(api.getGameDetail).mockReturnValue(new Promise(() => {}));
    vi.mocked(api.getGamePlayers).mockReturnValue(new Promise(() => {}));

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it("renders markets sorted by edge descending", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);

    const rows = await screen.findAllByTestId("market-row");
    expect(rows[0]).toHaveTextContent("h2h");
    expect(rows[1]).toHaveTextContent("spread");
  });

  it("renders player prop predictions", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);

    expect(await screen.findByText("points")).toBeInTheDocument();
    expect(screen.getByText("27.5")).toBeInTheDocument();
  });

  it("calls onClose when the close button is clicked", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);
    const onClose = vi.fn();

    render(<GameDetailModal gameId="g1" onClose={onClose} />);
    await screen.findByText(/BOS/);

    await userEvent.click(screen.getByRole("button", { name: /close/i }));
    expect(onClose).toHaveBeenCalled();
  });

  it("shows an error state when either fetch fails", async () => {
    vi.mocked(api.getGameDetail).mockRejectedValue(new Error("boom"));
    vi.mocked(api.getGamePlayers).mockResolvedValue([]);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);
    expect(await screen.findByText(/couldn't load game details/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test -- components/GameDetailModal.test.tsx`
Expected: FAIL — `Cannot find module './GameDetailModal'`.

- [ ] **Step 3: Write `frontend/src/components/GameDetailModal.tsx`**

```tsx
import { useEffect, useState } from "react";
import { api, type GameDetail, type PlayerProp } from "../api/client";

interface GameDetailModalProps {
  gameId: string;
  onClose: () => void;
}

export default function GameDetailModal({ gameId, onClose }: GameDetailModalProps) {
  const [detail, setDetail] = useState<GameDetail | null>(null);
  const [players, setPlayers] = useState<PlayerProp[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDetail(null);
    setPlayers(null);
    setError(null);
    Promise.all([api.getGameDetail(gameId), api.getGamePlayers(gameId)])
      .then(([detailResult, playersResult]) => {
        setDetail(detailResult);
        setPlayers(playersResult);
      })
      .catch(() => setError("Couldn't load game details."));
  }, [gameId]);

  const sortedMarkets = detail ? [...detail.markets].sort((a, b) => (b.edge ?? 0) - (a.edge ?? 0)) : [];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70" onClick={onClose}>
      <div
        className="max-h-[85vh] w-full max-w-2xl overflow-y-auto rounded-xl border border-[var(--color-nba-border)] bg-[var(--color-nba-900)] p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-bold">{detail ? `${detail.away_team} @ ${detail.home_team}` : "Game detail"}</h2>
          <button aria-label="Close" onClick={onClose} className="text-xl leading-none text-[var(--color-nba-text-dim)]">
            ×
          </button>
        </div>

        {error && <p className="text-[var(--color-loss)]">{error}</p>}
        {!error && !detail && <p>Loading…</p>}

        {detail?.prediction && (
          <p className="mb-4 text-sm text-[var(--color-nba-amber)]">
            Win probability: {Math.round(detail.prediction.home_win_probability * 100)}% · Margin:{" "}
            {detail.prediction.predicted_margin.toFixed(1)} · Total: {detail.prediction.predicted_total.toFixed(1)}
          </p>
        )}

        {sortedMarkets.length > 0 && (
          <table className="mb-4 w-full text-sm">
            <thead>
              <tr className="text-left text-[var(--color-nba-text-faint)]">
                <th>Market</th>
                <th>Selection</th>
                <th>Model %</th>
                <th>Market %</th>
                <th>Edge</th>
              </tr>
            </thead>
            <tbody>
              {sortedMarkets.map((market, i) => (
                <tr key={i} data-testid="market-row">
                  <td>{market.market}</td>
                  <td>{market.selection}</td>
                  <td>{Math.round(market.model_probability * 100)}%</td>
                  <td>{market.market_probability !== null ? `${Math.round(market.market_probability * 100)}%` : "—"}</td>
                  <td>{market.edge !== null ? `${(market.edge * 100).toFixed(1)}pp` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {players && players.length > 0 && (
          <ul className="text-sm">
            {players.map((player, i) => (
              <li key={i}>
                {player.player_name} — {player.stat}: {player.predicted_value}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Wire the modal into `frontend/src/pages/GamesPage.tsx`**

Replace the placeholder `{selectedGameId && ...}` line at the bottom of
`GamesPage` with:

```tsx
      {selectedGameId && (
        <GameDetailModal gameId={selectedGameId} onClose={() => setSelectedGameId(null)} />
      )}
```

And add the import at the top:

```tsx
import GameDetailModal from "../components/GameDetailModal";
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npm test -- components/GameDetailModal.test.tsx pages/GamesPage.test.tsx`
Expected: PASS (10 passed total)

- [ ] **Step 6: Commit**

```bash
cd "/Users/sigey/Documents/Projects/NBA_Predictor"
git add frontend/src/components/GameDetailModal.tsx frontend/src/components/GameDetailModal.test.tsx frontend/src/pages/GamesPage.tsx
git commit -m "feat: add game detail modal with markets and player props"
```

---

### Task 5: Data Hub — Team Hub and Player Hub panels

**Files:**
- Create: `frontend/src/components/TeamHubPanel.tsx`
- Create: `frontend/src/components/PlayerHubPanel.tsx`
- Test: `frontend/src/components/TeamHubPanel.test.tsx`, `frontend/src/components/PlayerHubPanel.test.tsx`

**Interfaces:**
- Consumes: `api.getHubTeams`, `api.getHubPlayers` (Task 2).
- Produces:
  - `TeamHubPanel()` — fetches `api.getHubTeams` on mount, groups rows by `conference` (East/West headers), each row sortable by clicking a column header (click toggles ascending/descending on `points_per_game` by default — implement generically: `sortKey` state defaulting to `"points_per_game"`). Loading/empty/error states as in Task 3.
  - `PlayerHubPanel()` — fetches `api.getHubPlayers`, renders a table (player, team, position, rating, PPG, RPG, APG, FG%, 3P%, FT%) sorted by `rating` descending by default, paginated 20 rows/page with "Next"/"Previous" buttons (`data-testid="player-hub-next"` / `"player-hub-prev"`).

- [ ] **Step 1: Write the failing tests**

```tsx
// frontend/src/components/TeamHubPanel.test.tsx
import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import TeamHubPanel from "./TeamHubPanel";
import { api } from "../api/client";

vi.mock("../api/client", () => ({ api: { getHubTeams: vi.fn() } }));
afterEach(() => vi.restoreAllMocks());

describe("TeamHubPanel", () => {
  it("groups teams by conference", async () => {
    vi.mocked(api.getHubTeams).mockResolvedValue([
      { abbreviation: "BOS", conference: "East", division: "Atlantic", wins: 10, losses: 2, points_per_game: 118, opp_points_per_game: 108, net_rating: 10, pace: 99, streak: 3 },
      { abbreviation: "LAL", conference: "West", division: "Pacific", wins: 8, losses: 4, points_per_game: 115, opp_points_per_game: 110, net_rating: 5, pace: 101, streak: -1 },
    ]);

    render(<TeamHubPanel />);

    await waitFor(() => expect(screen.getByText("BOS")).toBeInTheDocument());
    expect(screen.getByText("East")).toBeInTheDocument();
    expect(screen.getByText("West")).toBeInTheDocument();
  });

  it("shows an empty state when no team data is cached yet", async () => {
    vi.mocked(api.getHubTeams).mockResolvedValue([]);
    render(<TeamHubPanel />);
    await waitFor(() => expect(screen.getByText(/no team data/i)).toBeInTheDocument());
  });
});
```

```tsx
// frontend/src/components/PlayerHubPanel.test.tsx
import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import PlayerHubPanel from "./PlayerHubPanel";
import { api } from "../api/client";

vi.mock("../api/client", () => ({ api: { getHubPlayers: vi.fn() } }));
afterEach(() => vi.restoreAllMocks());

function makePlayer(i: number) {
  return {
    player_id: `${i}`, player_name: `Player ${i}`, team: "BOS", position: "G",
    rating: 100 - i, live_form_rating: 90, points_per_game: 20, rebounds_per_game: 5,
    assists_per_game: 4, fg_pct: 0.47, three_pt_pct: 0.37, ft_pct: 0.85, usage_rate: 0.24,
    minutes_per_game: 32,
  };
}

describe("PlayerHubPanel", () => {
  it("shows the first 20 players and paginates to the next page", async () => {
    vi.mocked(api.getHubPlayers).mockResolvedValue(Array.from({ length: 25 }, (_, i) => makePlayer(i)));

    render(<PlayerHubPanel />);

    await waitFor(() => expect(screen.getByText("Player 0")).toBeInTheDocument());
    expect(screen.queryByText("Player 20")).not.toBeInTheDocument();

    await userEvent.click(screen.getByTestId("player-hub-next"));
    expect(await screen.findByText("Player 20")).toBeInTheDocument();
  });

  it("shows an empty state when no player data is cached yet", async () => {
    vi.mocked(api.getHubPlayers).mockResolvedValue([]);
    render(<PlayerHubPanel />);
    await waitFor(() => expect(screen.getByText(/no player data/i)).toBeInTheDocument());
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test -- components/TeamHubPanel.test.tsx components/PlayerHubPanel.test.tsx`
Expected: FAIL — modules don't exist.

- [ ] **Step 3: Write `frontend/src/components/TeamHubPanel.tsx`**

```tsx
import { useEffect, useState } from "react";
import { api, type TeamHubRow } from "../api/client";

export default function TeamHubPanel() {
  const [rows, setRows] = useState<TeamHubRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getHubTeams().then(setRows).catch(() => setError("Couldn't load team data."));
  }, []);

  if (error) return <p className="text-[var(--color-loss)]">{error}</p>;
  if (rows === null) return <p>Loading teams…</p>;
  if (rows.length === 0) return <p>No team data cached yet.</p>;

  const conferences: Array<"East" | "West"> = ["East", "West"];

  return (
    <div className="space-y-6">
      {conferences.map((conference) => (
        <div key={conference}>
          <h3 className="mb-2 font-semibold">{conference}</h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-[var(--color-nba-text-faint)]">
                <th>Team</th>
                <th>W-L</th>
                <th>PPG</th>
                <th>Opp PPG</th>
                <th>Net Rtg</th>
                <th>Pace</th>
                <th>Streak</th>
              </tr>
            </thead>
            <tbody>
              {rows
                .filter((row) => row.conference === conference)
                .sort((a, b) => b.points_per_game - a.points_per_game)
                .map((row) => (
                  <tr key={row.abbreviation}>
                    <td>{row.abbreviation}</td>
                    <td>
                      {row.wins}-{row.losses}
                    </td>
                    <td>{row.points_per_game.toFixed(1)}</td>
                    <td>{row.opp_points_per_game.toFixed(1)}</td>
                    <td>{row.net_rating.toFixed(1)}</td>
                    <td>{row.pace.toFixed(1)}</td>
                    <td>{row.streak}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 4: Write `frontend/src/components/PlayerHubPanel.tsx`**

```tsx
import { useEffect, useState } from "react";
import { api, type PlayerHubRow } from "../api/client";

const PAGE_SIZE = 20;

export default function PlayerHubPanel() {
  const [rows, setRows] = useState<PlayerHubRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);

  useEffect(() => {
    api.getHubPlayers().then(setRows).catch(() => setError("Couldn't load player data."));
  }, []);

  if (error) return <p className="text-[var(--color-loss)]">{error}</p>;
  if (rows === null) return <p>Loading players…</p>;
  if (rows.length === 0) return <p>No player data cached yet.</p>;

  const sorted = [...rows].sort((a, b) => b.rating - a.rating);
  const pageRows = sorted.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const hasNext = (page + 1) * PAGE_SIZE < sorted.length;

  return (
    <div>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-[var(--color-nba-text-faint)]">
            <th>Player</th>
            <th>Team</th>
            <th>Pos</th>
            <th>Rating</th>
            <th>PPG</th>
            <th>RPG</th>
            <th>APG</th>
            <th>FG%</th>
            <th>3P%</th>
            <th>FT%</th>
          </tr>
        </thead>
        <tbody>
          {pageRows.map((row) => (
            <tr key={row.player_id}>
              <td>{row.player_name}</td>
              <td>{row.team}</td>
              <td>{row.position}</td>
              <td>{row.rating.toFixed(1)}</td>
              <td>{row.points_per_game.toFixed(1)}</td>
              <td>{row.rebounds_per_game.toFixed(1)}</td>
              <td>{row.assists_per_game.toFixed(1)}</td>
              <td>{(row.fg_pct * 100).toFixed(1)}</td>
              <td>{(row.three_pt_pct * 100).toFixed(1)}</td>
              <td>{(row.ft_pct * 100).toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-3 flex gap-2">
        <button data-testid="player-hub-prev" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
          Previous
        </button>
        <button data-testid="player-hub-next" disabled={!hasNext} onClick={() => setPage((p) => p + 1)}>
          Next
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npm test -- components/TeamHubPanel.test.tsx components/PlayerHubPanel.test.tsx`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
cd "/Users/sigey/Documents/Projects/NBA_Predictor"
git add frontend/src/components/TeamHubPanel.tsx frontend/src/components/TeamHubPanel.test.tsx frontend/src/components/PlayerHubPanel.tsx frontend/src/components/PlayerHubPanel.test.tsx
git commit -m "feat: add Team Hub and Player Hub panels"
```

---

### Task 6: Data Hub — Power Rankings, Standings, Track Record panels

**Files:**
- Create: `frontend/src/components/PowerRankingsPanel.tsx`
- Create: `frontend/src/components/StandingsPanel.tsx`
- Create: `frontend/src/components/TrackRecordPanel.tsx`
- Test: `frontend/src/components/PowerRankingsPanel.test.tsx`, `frontend/src/components/StandingsPanel.test.tsx`, `frontend/src/components/TrackRecordPanel.test.tsx`

**Interfaces:**
- Consumes: `api.getHubRankings`, `api.getHubStandings`, `api.getTrackRecord` (Task 2).
- Produces:
  - `PowerRankingsPanel()` — ordered list by `rank`, showing `abbreviation`, `power_rating`, and a trend glyph (`"▲"`/`"▼"`/`"–"` for `up`/`down`/`steady`).
  - `StandingsPanel()` — two columns (East/West), each an ordered table by `seed` showing `abbreviation`, `wins`-`losses`, `win_pct`, `games_back`, and `playoff_status` — this is the spec's "predicted finishing positions" requirement.
  - `TrackRecordPanel()` — a table of `market`, `total_predictions`, `correct_predictions`, `hit_rate` (as a percentage).
  - All three share Task 3's loading/empty/error state convention.

- [ ] **Step 1: Write the failing tests**

```tsx
// frontend/src/components/PowerRankingsPanel.test.tsx
import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import PowerRankingsPanel from "./PowerRankingsPanel";
import { api } from "../api/client";

vi.mock("../api/client", () => ({ api: { getHubRankings: vi.fn() } }));
afterEach(() => vi.restoreAllMocks());

describe("PowerRankingsPanel", () => {
  it("renders ranked teams with trend glyphs", async () => {
    vi.mocked(api.getHubRankings).mockResolvedValue([
      { rank: 1, abbreviation: "BOS", power_rating: 1620, trend: "up" },
      { rank: 2, abbreviation: "LAL", power_rating: 1590, trend: "down" },
    ]);

    render(<PowerRankingsPanel />);

    await waitFor(() => expect(screen.getByText("BOS")).toBeInTheDocument());
    expect(screen.getByText("▲")).toBeInTheDocument();
    expect(screen.getByText("▼")).toBeInTheDocument();
  });

  it("shows an empty state when no rankings are cached yet", async () => {
    vi.mocked(api.getHubRankings).mockResolvedValue([]);
    render(<PowerRankingsPanel />);
    await waitFor(() => expect(screen.getByText(/no ranking data/i)).toBeInTheDocument());
  });
});
```

```tsx
// frontend/src/components/StandingsPanel.test.tsx
import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import StandingsPanel from "./StandingsPanel";
import { api } from "../api/client";

vi.mock("../api/client", () => ({ api: { getHubStandings: vi.fn() } }));
afterEach(() => vi.restoreAllMocks());

describe("StandingsPanel", () => {
  it("splits standings into East and West columns by seed", async () => {
    vi.mocked(api.getHubStandings).mockResolvedValue([
      { conference: "East", seed: 1, abbreviation: "BOS", wins: 50, losses: 20, win_pct: 0.714, games_back: 0, playoff_status: "clinched" },
      { conference: "West", seed: 1, abbreviation: "LAL", wins: 48, losses: 22, win_pct: 0.686, games_back: 0, playoff_status: "clinched" },
    ]);

    render(<StandingsPanel />);

    await waitFor(() => expect(screen.getByText("BOS")).toBeInTheDocument());
    expect(screen.getByText("East")).toBeInTheDocument();
    expect(screen.getByText("West")).toBeInTheDocument();
  });

  it("shows an empty state when no standings are cached yet", async () => {
    vi.mocked(api.getHubStandings).mockResolvedValue([]);
    render(<StandingsPanel />);
    await waitFor(() => expect(screen.getByText(/no standings data/i)).toBeInTheDocument());
  });
});
```

```tsx
// frontend/src/components/TrackRecordPanel.test.tsx
import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import TrackRecordPanel from "./TrackRecordPanel";
import { api } from "../api/client";

vi.mock("../api/client", () => ({ api: { getTrackRecord: vi.fn() } }));
afterEach(() => vi.restoreAllMocks());

describe("TrackRecordPanel", () => {
  it("renders hit rate as a percentage", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      { market: "h2h", total_predictions: 100, correct_predictions: 58, hit_rate: 0.58 },
    ]);

    render(<TrackRecordPanel />);

    await waitFor(() => expect(screen.getByText("h2h")).toBeInTheDocument());
    expect(screen.getByText("58%")).toBeInTheDocument();
  });

  it("shows an empty state when no predictions are tracked yet", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([]);
    render(<TrackRecordPanel />);
    await waitFor(() => expect(screen.getByText(/no tracked predictions/i)).toBeInTheDocument());
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test -- components/PowerRankingsPanel.test.tsx components/StandingsPanel.test.tsx components/TrackRecordPanel.test.tsx`
Expected: FAIL — modules don't exist.

- [ ] **Step 3: Write `frontend/src/components/PowerRankingsPanel.tsx`**

```tsx
import { useEffect, useState } from "react";
import { api, type PowerRankingRow } from "../api/client";

const TREND_GLYPH: Record<PowerRankingRow["trend"], string> = { up: "▲", down: "▼", steady: "–" };

export default function PowerRankingsPanel() {
  const [rows, setRows] = useState<PowerRankingRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getHubRankings().then(setRows).catch(() => setError("Couldn't load rankings."));
  }, []);

  if (error) return <p className="text-[var(--color-loss)]">{error}</p>;
  if (rows === null) return <p>Loading rankings…</p>;
  if (rows.length === 0) return <p>No ranking data cached yet.</p>;

  return (
    <ol className="space-y-1 text-sm">
      {[...rows]
        .sort((a, b) => a.rank - b.rank)
        .map((row) => (
          <li key={row.abbreviation} className="flex justify-between border-b border-[var(--color-nba-border)] py-1">
            <span>
              #{row.rank} {row.abbreviation}
            </span>
            <span>
              {row.power_rating.toFixed(0)} {TREND_GLYPH[row.trend]}
            </span>
          </li>
        ))}
    </ol>
  );
}
```

- [ ] **Step 4: Write `frontend/src/components/StandingsPanel.tsx`**

```tsx
import { useEffect, useState } from "react";
import { api, type StandingsRow } from "../api/client";

export default function StandingsPanel() {
  const [rows, setRows] = useState<StandingsRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getHubStandings().then(setRows).catch(() => setError("Couldn't load standings."));
  }, []);

  if (error) return <p className="text-[var(--color-loss)]">{error}</p>;
  if (rows === null) return <p>Loading standings…</p>;
  if (rows.length === 0) return <p>No standings data cached yet.</p>;

  const conferences: Array<"East" | "West"> = ["East", "West"];

  return (
    <div className="grid gap-6 sm:grid-cols-2">
      {conferences.map((conference) => (
        <div key={conference}>
          <h3 className="mb-2 font-semibold">{conference}</h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-[var(--color-nba-text-faint)]">
                <th>Seed</th>
                <th>Team</th>
                <th>W-L</th>
                <th>GB</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {rows
                .filter((row) => row.conference === conference)
                .sort((a, b) => a.seed - b.seed)
                .map((row) => (
                  <tr key={row.abbreviation}>
                    <td>{row.seed}</td>
                    <td>{row.abbreviation}</td>
                    <td>
                      {row.wins}-{row.losses}
                    </td>
                    <td>{row.games_back.toFixed(1)}</td>
                    <td>{row.playoff_status}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 5: Write `frontend/src/components/TrackRecordPanel.tsx`**

```tsx
import { useEffect, useState } from "react";
import { api, type TrackRecord } from "../api/client";

export default function TrackRecordPanel() {
  const [rows, setRows] = useState<TrackRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getTrackRecord().then(setRows).catch(() => setError("Couldn't load track record."));
  }, []);

  if (error) return <p className="text-[var(--color-loss)]">{error}</p>;
  if (rows === null) return <p>Loading track record…</p>;
  if (rows.length === 0) return <p>No tracked predictions yet.</p>;

  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-left text-[var(--color-nba-text-faint)]">
          <th>Market</th>
          <th>Predictions</th>
          <th>Correct</th>
          <th>Hit Rate</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.market}>
            <td>{row.market}</td>
            <td>{row.total_predictions}</td>
            <td>{row.correct_predictions}</td>
            <td>{Math.round(row.hit_rate * 100)}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `npm test -- components/PowerRankingsPanel.test.tsx components/StandingsPanel.test.tsx components/TrackRecordPanel.test.tsx`
Expected: PASS (6 passed)

- [ ] **Step 7: Commit**

```bash
cd "/Users/sigey/Documents/Projects/NBA_Predictor"
git add frontend/src/components/PowerRankingsPanel.tsx frontend/src/components/PowerRankingsPanel.test.tsx frontend/src/components/StandingsPanel.tsx frontend/src/components/StandingsPanel.test.tsx frontend/src/components/TrackRecordPanel.tsx frontend/src/components/TrackRecordPanel.test.tsx
git commit -m "feat: add Power Rankings, Standings, and Track Record panels"
```

---

### Task 7: Data Hub page shell (tabs) and Model Summary page

**Files:**
- Create: `frontend/src/pages/DataHubPage.tsx`
- Create: `frontend/src/pages/ModelSummaryPage.tsx`
- Test: `frontend/src/pages/DataHubPage.test.tsx`, `frontend/src/pages/ModelSummaryPage.test.tsx`

**Interfaces:**
- Consumes: all five panels from Tasks 5-6; a new `api.getManifest(): Promise<Manifest>` added to `api/client.ts` in this task (`Manifest` type: `{ model_version: string; trained_at: string; models: string[]; metrics: Record<string, Record<string, number | null>> }`, matching Phase 3's `models.manifest.build_manifest` shape, served by a new backend route — see Self-Review).
- Produces:
  - `DataHubPage()` — five tab buttons (`Team Hub`, `Player Hub`, `Power Rankings`, `Standings`, `Track Record`), `useState` tracks the active tab, renders exactly one panel at a time; clicking a tab button switches the active panel (`data-testid="hub-tab-{name}"` on each button, kebab-case `name`).
  - `ModelSummaryPage()` — fetches `api.getManifest()`, renders `model_version`, `trained_at`, the list of `models`, and a metrics table (one row per model, one column per metric key found in that model's metrics dict).

- [ ] **Step 1: Add `getManifest` to `frontend/src/api/client.ts`**

```ts
export interface Manifest {
  model_version: string;
  trained_at: string;
  models: string[];
  metrics: Record<string, Record<string, number | null>>;
}
```

Add `getManifest: () => fetchJson<Manifest>("/manifest"),` to the `api`
object. (This assumes a `GET /manifest` route that reads
`models/manifest.json` off disk — add it to
`src/nba_predictor/api/routes.py` as part of this task too, reusing the
existing `deps.get_models_dir` dependency from Phase 4 Task 9:

```python
import json

from nba_predictor.api.deps import get_models_dir


@router.get("/manifest")
def get_manifest(models_dir: Path = Depends(get_models_dir)) -> dict:
    manifest_path = models_dir / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="No manifest found — run /retrain first")
    return json.loads(manifest_path.read_text())
```

Add a corresponding backend test to `tests/test_api_admin.py` alongside the
existing admin tests:

```python
def test_get_manifest_404_when_absent(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, public_mode=False)
    response = client.get("/manifest")
    assert response.status_code == 404


def test_get_manifest_returns_file_contents(tmp_path, monkeypatch):
    import json

    client = _client(tmp_path, monkeypatch, public_mode=False)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "manifest.json").write_text(json.dumps({"model_version": "v1", "trained_at": "t", "models": [], "metrics": {}}))

    response = client.get("/manifest")
    assert response.status_code == 200
    assert response.json()["model_version"] == "v1"
```

Run `pytest tests/test_api_admin.py -v` — expect PASS (7 passed, 5 existing + 2 new) — before continuing to the frontend steps.)

- [ ] **Step 2: Write the failing frontend tests**

```tsx
// frontend/src/pages/DataHubPage.test.tsx
import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import DataHubPage from "./DataHubPage";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    getHubTeams: vi.fn().mockResolvedValue([]),
    getHubPlayers: vi.fn().mockResolvedValue([]),
    getHubRankings: vi.fn().mockResolvedValue([]),
    getHubStandings: vi.fn().mockResolvedValue([]),
    getTrackRecord: vi.fn().mockResolvedValue([]),
  },
}));

afterEach(() => vi.restoreAllMocks());

describe("DataHubPage", () => {
  it("shows the Team Hub panel by default", async () => {
    render(<DataHubPage />);
    await waitFor(() => expect(screen.getByText(/no team data/i)).toBeInTheDocument());
  });

  it("switches to the Player Hub panel when its tab is clicked", async () => {
    render(<DataHubPage />);
    await userEvent.click(screen.getByTestId("hub-tab-player-hub"));
    await waitFor(() => expect(screen.getByText(/no player data/i)).toBeInTheDocument());
  });

  it("switches to the Track Record panel when its tab is clicked", async () => {
    render(<DataHubPage />);
    await userEvent.click(screen.getByTestId("hub-tab-track-record"));
    await waitFor(() => expect(screen.getByText(/no tracked predictions/i)).toBeInTheDocument());
  });
});
```

```tsx
// frontend/src/pages/ModelSummaryPage.test.tsx
import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import ModelSummaryPage from "./ModelSummaryPage";
import { api } from "../api/client";

vi.mock("../api/client", () => ({ api: { getManifest: vi.fn() } }));
afterEach(() => vi.restoreAllMocks());

describe("ModelSummaryPage", () => {
  it("renders model version and metrics", async () => {
    vi.mocked(api.getManifest).mockResolvedValue({
      model_version: "v20261101120000",
      trained_at: "2026-11-01T12:00:00",
      models: ["win_probability", "margin", "total"],
      metrics: { win_probability: { accuracy: 0.64 } },
    });

    render(<ModelSummaryPage />);

    await waitFor(() => expect(screen.getByText("v20261101120000")).toBeInTheDocument());
    expect(screen.getByText("win_probability")).toBeInTheDocument();
    expect(screen.getByText("0.64")).toBeInTheDocument();
  });

  it("shows an empty state when no model has been trained yet", async () => {
    vi.mocked(api.getManifest).mockRejectedValue(new Error("404"));
    render(<ModelSummaryPage />);
    await waitFor(() => expect(screen.getByText(/no model has been trained/i)).toBeInTheDocument());
  });
});
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `npm test -- pages/DataHubPage.test.tsx pages/ModelSummaryPage.test.tsx`
Expected: FAIL — modules don't exist.

- [ ] **Step 4: Write `frontend/src/pages/DataHubPage.tsx`**

```tsx
import { useState } from "react";
import TeamHubPanel from "../components/TeamHubPanel";
import PlayerHubPanel from "../components/PlayerHubPanel";
import PowerRankingsPanel from "../components/PowerRankingsPanel";
import StandingsPanel from "../components/StandingsPanel";
import TrackRecordPanel from "../components/TrackRecordPanel";

const TABS = [
  { key: "team-hub", label: "Team Hub", panel: TeamHubPanel },
  { key: "player-hub", label: "Player Hub", panel: PlayerHubPanel },
  { key: "power-rankings", label: "Power Rankings", panel: PowerRankingsPanel },
  { key: "standings", label: "Standings", panel: StandingsPanel },
  { key: "track-record", label: "Track Record", panel: TrackRecordPanel },
] as const;

export default function DataHubPage() {
  const [activeTab, setActiveTab] = useState<(typeof TABS)[number]["key"]>("team-hub");
  const ActivePanel = TABS.find((tab) => tab.key === activeTab)!.panel;

  return (
    <div>
      <div className="mb-4 flex gap-2 border-b border-[var(--color-nba-border)]">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            data-testid={`hub-tab-${tab.key}`}
            onClick={() => setActiveTab(tab.key)}
            className={`px-3 py-2 text-sm ${
              activeTab === tab.key ? "border-b-2 border-[var(--color-nba-orange)] font-semibold" : "text-[var(--color-nba-text-dim)]"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <ActivePanel />
    </div>
  );
}
```

- [ ] **Step 5: Write `frontend/src/pages/ModelSummaryPage.tsx`**

```tsx
import { useEffect, useState } from "react";
import { api, type Manifest } from "../api/client";

export default function ModelSummaryPage() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [notTrained, setNotTrained] = useState(false);

  useEffect(() => {
    api
      .getManifest()
      .then(setManifest)
      .catch(() => setNotTrained(true));
  }, []);

  if (notTrained) return <p>No model has been trained yet.</p>;
  if (!manifest) return <p>Loading model summary…</p>;

  return (
    <div>
      <p className="mb-1 text-sm text-[var(--color-nba-text-dim)]">Model version</p>
      <p className="mb-4 font-mono">{manifest.model_version}</p>
      <p className="mb-1 text-sm text-[var(--color-nba-text-dim)]">Trained at</p>
      <p className="mb-4 font-mono">{manifest.trained_at}</p>

      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-[var(--color-nba-text-faint)]">
            <th>Model</th>
            <th>Metrics</th>
          </tr>
        </thead>
        <tbody>
          {manifest.models.map((modelName) => (
            <tr key={modelName}>
              <td>{modelName}</td>
              <td>
                {Object.entries(manifest.metrics[modelName] ?? {})
                  .map(([key, value]) => `${key}: ${value}`)
                  .join(", ")}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `npm test -- pages/DataHubPage.test.tsx pages/ModelSummaryPage.test.tsx`
Expected: PASS (5 passed)

- [ ] **Step 7: Commit**

```bash
cd "/Users/sigey/Documents/Projects/NBA_Predictor"
git add frontend/src/pages/DataHubPage.tsx frontend/src/pages/DataHubPage.test.tsx frontend/src/pages/ModelSummaryPage.tsx frontend/src/pages/ModelSummaryPage.test.tsx frontend/src/api/client.ts src/nba_predictor/api/routes.py tests/test_api_admin.py
git commit -m "feat: add Data Hub tabs and Model Summary page, plus /manifest route"
```

---

### Task 8: App routing and navigation

**Files:**
- Modify: `frontend/src/App.tsx`
- Test: `frontend/src/App.test.tsx` (extend)

**Interfaces:**
- Consumes: `GamesPage`, `DataHubPage`, `ModelSummaryPage` (Tasks 3, 7).
- Produces: `App` renders a `<nav>` with three links (`Games` → `/`, `Data Hub` → `/hub`, `Model Summary` → `/model`) and a `<Routes>` block mapping each path to its page component.

- [ ] **Step 1: Extend `frontend/src/App.test.tsx`**

```tsx
// append to frontend/src/App.test.tsx
import userEvent from "@testing-library/user-event";

describe("App routing", () => {
  it("navigates to the Data Hub page when its nav link is clicked", async () => {
    render(
      <MemoryRouter>
        <App />
      </MemoryRouter>
    );

    await userEvent.click(screen.getByRole("link", { name: "Data Hub" }));
    expect(await screen.findByTestId("hub-tab-team-hub")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- App.test.tsx`
Expected: FAIL — no `<nav>` links exist yet.

- [ ] **Step 3: Write the final `frontend/src/App.tsx`**

```tsx
import { NavLink, Route, Routes } from "react-router-dom";
import GamesPage from "./pages/GamesPage";
import DataHubPage from "./pages/DataHubPage";
import ModelSummaryPage from "./pages/ModelSummaryPage";

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  `px-3 py-2 text-sm ${isActive ? "font-semibold text-[var(--color-nba-orange)]" : "text-[var(--color-nba-text-dim)]"}`;

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="flex items-center justify-between border-b border-[var(--color-nba-border)] px-6 py-4">
        <h1 className="text-xl font-bold tracking-tight">NBA Predictor</h1>
        <nav className="flex gap-2">
          <NavLink to="/" end className={navLinkClass}>
            Games
          </NavLink>
          <NavLink to="/hub" className={navLinkClass}>
            Data Hub
          </NavLink>
          <NavLink to="/model" className={navLinkClass}>
            Model Summary
          </NavLink>
        </nav>
      </header>
      <main className="px-6 py-8">
        <Routes>
          <Route path="/" element={<GamesPage />} />
          <Route path="/hub" element={<DataHubPage />} />
          <Route path="/model" element={<ModelSummaryPage />} />
        </Routes>
      </main>
    </div>
  );
}
```

- [ ] **Step 4: Run the full frontend test suite**

Run: `npm test`
Expected: all tests across every task pass.

- [ ] **Step 5: Verify the production build still works**

```bash
npm run build
```

Expected: exits 0.

- [ ] **Step 6: Commit**

```bash
cd "/Users/sigey/Documents/Projects/NBA_Predictor"
git add frontend/src/App.tsx frontend/src/App.test.tsx
git commit -m "feat: add app navigation and routing"
```

---

## Self-Review Notes

- **Spec coverage:** Fixtures/Games page with per-game modal (markets +
  player props), Data Hub with Team Hub (grouped by conference)/Player
  Hub/Power Rankings/Standings ("predicted finishing positions")/Track
  Record tabs, Model Summary page, and the navy/orange Tailwind v4 dark
  theme — every frontend bullet in spec §6 has a task. The private
  Calibration page from spec §6 is **not** included here — it depends on
  Phase 4's `/calibration` route, which wasn't built in Phase 4 (only
  `/hub/track-record` was); flagged as a follow-up "Phase 5.5" task once a
  `/calibration` backend route exists.
- **Cross-phase dependency introduced:** Task 7 adds a `GET /manifest`
  backend route (small, additive, tested) since the frontend's Model
  Summary page needs it and Phase 4 didn't add it — this is called out
  explicitly rather than silently reaching into Phase 4's territory.
- **Placeholder scan:** no TBD/TODO; every step has runnable code.
- **Type consistency:** `Game`/`GameDetail`/`MarketPrediction`/`PlayerProp`/
  `TrackRecord`/`TeamHubRow`/`PlayerHubRow`/`PowerRankingRow`/`StandingsRow`/
  `Manifest` field names match between `api/client.ts` (Task 2, extended in
  Task 7) and every component/page that consumes them.
