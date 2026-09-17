import { useEffect, useState } from "react";
import { api, type CalibrationBin } from "../api/client";

export default function CalibrationPage() {
  const [bins, setBins] = useState<CalibrationBin[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getCalibration()
      .then(setBins)
      .catch(() => setError("Couldn't load calibration data."));
  }, []);

  if (error) return <p className="text-[var(--color-shotclock)]">{error}</p>;
  if (bins === null) return <p>Loading calibration…</p>;
  if (bins.length === 0) return <p>No settled predictions yet to calibrate against.</p>;

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
