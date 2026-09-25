import { useEffect, useState } from "react";
import { api, type CalibrationBin } from "../api/client";

export default function CalibrationPage() {
  const [bins, setBins] = useState<CalibrationBin[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    setError(null);
    setBins(null);
    api
      .getCalibration()
      .then(setBins)
      .catch(() => setError("calibration"));
  }, [reloadKey]);

  if (error)
    return (
      <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded border border-[var(--color-line)] p-3 text-sm">
        <span>We couldn't load calibration data. Check your connection and try again.</span>
        <button onClick={() => setReloadKey((k) => k + 1)} className="rounded border border-[var(--color-line)] px-3 py-1 text-xs font-semibold hover:border-[var(--color-hardwood)]">Try again</button>
      </div>
    );
  if (bins === null) return <p role="status" aria-live="polite">Loading calibration…</p>;
  if (bins.length === 0) return <p>Not enough finished games to check calibration yet. This fills in as the season is played.</p>;

  return (
    <div>
      <p className="mb-4 max-w-prose text-sm text-[var(--color-net-dim)]">
        For each probability range the model has predicted, how often the home team actually won. A
        well-calibrated model's predicted rate and actual rate should track closely.
      </p>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-[var(--color-net-faint)]">
            <th>Predicted range</th>
            <th>Predicted rate</th>
            <th>Actual rate</th>
            <th>Games</th>
          </tr>
        </thead>
        <tbody>
          {bins.map((bin) => (
            <tr key={bin.bin_start} data-testid="calibration-row">
              <td>
                {Math.round(bin.bin_start * 100)}–{Math.round(bin.bin_end * 100)}%
              </td>
              <td>{Math.round(bin.predicted_rate * 100)}%</td>
              <td>{Math.round(bin.actual_rate * 100)}%</td>
              <td>{bin.count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
