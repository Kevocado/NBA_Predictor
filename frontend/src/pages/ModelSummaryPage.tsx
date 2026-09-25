import { useEffect, useState } from "react";
import { api, type Manifest } from "../api/client";
import { EmptyState, Skeleton, modelDate, stat } from "../predictor-ui";

const MODEL_NAMES: Record<string, string> = {
  win_probability: "Win probability",
  margin: "Margin",
  total: "Total points",
};
const METRIC_NAMES: Record<string, string> = {
  accuracy: "Accuracy",
  log_loss: "Log loss",
  brier: "Brier score",
  brier_score: "Brier score",
  mae: "Mean abs. error",
  rmse: "RMSE",
  auc: "AUC",
};

// Metrics are fractions or points; three significant places read cleanly.
const metric = (v: number | null) => (v === null || !Number.isFinite(v) ? "—" : Math.abs(v) < 10 ? String(+v.toFixed(3)) : stat(v));

function trainedOn(iso: string): string | null {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

export default function ModelSummaryPage() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [notTrained, setNotTrained] = useState(false);

  useEffect(() => {
    api
      .getManifest()
      .then((m) => (Array.isArray(m?.models) ? setManifest(m) : setNotTrained(true)))
      .catch(() => setNotTrained(true));
  }, []);

  if (notTrained) return <EmptyState message="No model has been trained yet." />;
  if (!manifest) return <Skeleton label="Loading model summary…" />;
  const trained = trainedOn(manifest.trained_at);

  return (
    <div>
      <h2 className="font-pr-display text-2xl font-bold uppercase tracking-wide">{modelDate(manifest.model_version)}</h2>
      {trained && <p className="mb-4 text-sm text-pr-text-dim">Trained {trained}</p>}

      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-pr-text-dim">
            <th>Model</th>
            <th>Metrics</th>
          </tr>
        </thead>
        <tbody>
          {manifest.models.map((modelName) => (
            <tr key={modelName}>
              <td>{MODEL_NAMES[modelName] ?? modelName}</td>
              <td className="flex flex-wrap gap-x-4 gap-y-1">
                {Object.entries(manifest.metrics?.[modelName] ?? {}).map(([key, value]) => (
                  <span key={key}>
                    <span className="text-pr-text-dim">{METRIC_NAMES[key] ?? key}</span> <span>{metric(value)}</span>
                  </span>
                ))}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
