import { NavLink, Route, Routes } from "react-router-dom";
import GamesPage from "./pages/GamesPage";
import DataHubPage from "./pages/DataHubPage";
import ModelSummaryPage from "./pages/ModelSummaryPage";
import CalibrationPage from "./pages/CalibrationPage";

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  `border-b-2 px-1 py-1 text-sm transition-colors ${
    isActive
      ? "border-[var(--color-hardwood)] text-[var(--color-net)]"
      : "border-transparent text-[var(--color-net-dim)] hover:text-[var(--color-net)]"
  }`;

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-[var(--color-line)]">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-4 px-6 py-4">
          <h1 className="text-2xl tracking-tight">NBA Predictor</h1>
          <nav className="flex gap-6">
            <NavLink to="/" end className={navLinkClass}>
              Games
            </NavLink>
            <NavLink to="/hub" className={navLinkClass}>
              Data Hub
            </NavLink>
            <NavLink to="/model" className={navLinkClass}>
              Model Summary
            </NavLink>
            <NavLink to="/calibration" className={navLinkClass}>
              Calibration
            </NavLink>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-5xl px-6 py-8">
        <Routes>
          <Route path="/" element={<GamesPage />} />
          <Route path="/hub" element={<DataHubPage />} />
          <Route path="/model" element={<ModelSummaryPage />} />
          <Route path="/calibration" element={<CalibrationPage />} />
        </Routes>
      </main>
    </div>
  );
}