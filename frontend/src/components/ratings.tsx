import { rankTier } from "../predictor-ui";
import type { RatingFigure, RatingMetric } from "../api/client";

/**
 * The pieces a rating is made of, in one place.
 *
 * The game detail draws the ratings as a table -- the away side, the metric,
 * the home side -- and the Teams page draws the same eleven metrics for one
 * team as a list. Both read the same figures, the same rank chips and the same
 * colour scale, so they share these pieces rather than each carrying its own.
 */

/** The metric's name, with its unit under it. */
export function RatingLabel({ metric }: { metric: RatingMetric }) {
  return (
    <>
      {metric.label}
      {metric.unit && (
        <span className="block text-xs uppercase tracking-wide text-pr-text-faint">{metric.unit}</span>
      )}
    </>
  );
}

/** One team's figure for one metric, and its place in the league. */
export function RatingValue({
  figure,
  n,
  label,
  team,
}: {
  figure?: RatingFigure | null;
  n: number;
  label: string;
  team?: string;
}) {
  if (!figure || figure.value === null) {
    // No figure is no row of numbers: an em dash, never a zero.
    return (
      <span className="pr-matchup-missing" data-testid="rating-missing">
        —
      </span>
    );
  }
  const tier = figure.rank ? rankTier(figure.rank, n) : "bad";
  return (
    <span className="inline-flex items-center justify-center gap-1.5">
      <span className="pr-rating-value" data-testid="rating-value">
        {figure.value}
      </span>
      <span
        className={`pr-rank-box pr-rank-${tier} pr-rating-rank`}
        data-testid="rating-rank"
        aria-label={`${team ? `${team}: ` : ""}${figure.rank} of ${n} for ${label}`}
      >
        {figure.rank}
      </span>
    </span>
  );
}

/**
 * One team's ratings, one line per metric: the metric, the figure, the rank.
 *
 * The same eleven metrics the game detail shows, read from the same payload.
 */
export function RatingList({
  metrics,
  figure,
  n,
  team,
  testId,
}: {
  metrics: RatingMetric[];
  figure: (metric: RatingMetric) => RatingFigure | null | undefined;
  n: number;
  team?: string;
  testId?: string;
}) {
  return (
    <ul className="text-sm" data-testid={testId}>
      {metrics.map((metric) => (
        <li
          key={metric.key}
          className="flex items-center justify-between gap-2"
          data-testid={`rating-row-${metric.key}`}
        >
          <span>
            <RatingLabel metric={metric} />
          </span>
          <RatingValue figure={figure(metric)} n={n} label={metric.label} team={team} />
        </li>
      ))}
    </ul>
  );
}
