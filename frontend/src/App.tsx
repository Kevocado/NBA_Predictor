import { Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { AppFrame } from "./predictor-ui";
import { SITES } from "./lib/sites";
import GamesPage from "./pages/GamesPage";
import DataHubPage from "./pages/DataHubPage";
import ModelSummaryPage from "./pages/ModelSummaryPage";
import CalibrationPage from "./pages/CalibrationPage";

// Page tabs map onto the existing routes, so deep links and refreshes keep working.
const TABS = [
  { id: "/", label: "Games" },
  { id: "/hub", label: "Data Hub" },
  { id: "/calibration-report", label: "Calibration" },
  { id: "/model", label: "Model" },
];

export default function App() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const active = TABS.find((tab) => tab.id !== "/" && pathname.startsWith(tab.id))?.id ?? "/";

  return (
    <AppFrame sport="nba" sportName="NBA" sites={SITES} tabs={TABS} activeTab={active} onTab={(id) => navigate(id)}>
      <Routes>
        <Route path="/" element={<GamesPage />} />
        <Route path="/hub" element={<DataHubPage />} />
        <Route path="/model" element={<ModelSummaryPage />} />
        <Route path="/calibration-report" element={<CalibrationPage />} />
      </Routes>
    </AppFrame>
  );
}
