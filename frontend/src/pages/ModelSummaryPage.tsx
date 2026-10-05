import { useEffect, useState } from "react";
import { api, type Manifest, type PlayerPropsManifest } from "../api/client";
import { EmptyState, Skeleton, modelDate, stat } from "../predictor-ui";

const FALLBACK_COPY = "Training data not published yet — it appears after the next retrain.";

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
  if (Number.isNaN(d.getTime())) return null;
  const part = (o: Intl.DateTimeFormatOptions) => d.toLocaleDateString("en-US", o);
  return `${part({ day: "numeric" })} ${part({ month: "short" })} ${part({ year: "numeric" })}`;
}

export default function ModelSummaryPage() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [propsManifest, setPropsManifest] = useState<PlayerPropsManifest | null>(null);
  const [notTrained, setNotTrained] = useState(false);

  useEffect(() => {
    api
      .getManifest()
      .then((m) => (Array.isArray(m?.models) ? setManifest(m) : setNotTrained(true)))
      .catch(() => setNotTrained(true));
    api
      .getPlayerPropsManifest()
      .then(setPropsManifest)
      .catch(() => setPropsManifest(null));
  }, []);

  if (notTrained) return <EmptyState message="No model has been trained yet." />;
  if (!manifest) return <Skeleton label="Loading model summary…" />;
  const trained = trainedOn(manifest.trained_at);

  const training = manifest.training;
  const propsTraining = propsManifest?.training;

  return (
    <div>
      <h2 className="font-pr-display text-2xl font-bold uppercase tracking-wide">{modelDate(manifest.model_version)}</h2>
      {trained && <p className="mb-4 text-sm text-pr-text-dim">Trained {trained}</p>}

      <div className="overflow-x-auto">

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
              <td>
                <div className="flex flex-wrap gap-x-4 gap-y-1">
                  {Object.entries(manifest.metrics?.[modelName] ?? {}).map(([key, value]) => (
                    <span key={key}>
                      <span className="text-pr-text-dim">{METRIC_NAMES[key] ?? key}</span> <span>{metric(value)}</span>
                    </span>
                  ))}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
        </tbody>
        </table>
      </div>

      <h2 className="mb-2 mt-6 font-pr-display text-lg font-semibold uppercase tracking-wide">Training data</h2>
      {training?.n_train_games != null ? (
        <div className="text-sm">
          <p className="mb-1">
            <span className="text-pr-text-dim">Training matches: </span>
            <span>{training.n_train_games.toLocaleString()}</span>
          </p>
          {training.n_current_season_games != null && (
            <p className="mb-1">
              <span className="text-pr-text-dim">Current-season matches: </span>
              <span>{training.n_current_season_games.toLocaleString()}</span>
            </p>
          )}
          {training.n_holdout_games != null && (
            <p className="mb-1">
              <span className="text-pr-text-dim">Held-out validation matches: </span>
              <span>{training.n_holdout_games.toLocaleString()}</span>
            </p>
          )}
          <p className="mt-2 text-pr-text-dim">
            Current-season matches count only once their stats are final and the model has
            retrained.
          </p>
        </div>
      ) : (
        <p className="text-sm text-pr-text-dim">{FALLBACK_COPY}</p>
      )}

      <h3 className="mb-2 mt-4 font-pr-display text-base font-semibold uppercase tracking-wide">Player props</h3>
      {propsTraining?.n_train_player_games != null ? (
        <p className="text-sm">
          <span className="text-pr-text-dim">Player-games used: </span>
          <span>{propsTraining.n_train_player_games.toLocaleString()}</span>
          {propsTraining.in_sample_metrics && (
            <span className="text-pr-text-dim"> (in-sample)</span>
          )}
        </p>
      ) : (
        <p className="text-sm text-pr-text-dim">{FALLBACK_COPY}</p>
      )}
    </div>
  );
}
