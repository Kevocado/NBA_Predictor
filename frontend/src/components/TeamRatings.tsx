import type { GameRatings, RatingMetric } from "../api/client";
import TeamLogo from "./TeamLogo";
import { RatingLabel, RatingValue } from "./ratings";

/**
 * The two sides' ratings against the league.
 *
 * Every row is one metric and both teams' figures for it, each with its place
 * in the league. The figure comes first and the rank beside it: "112.3, 3rd of
 * 30" is a claim a reader can check, and the chip's colour says the same thing
 * again -- the kit's `rankTier`, so the colour scale is the one the rest of the
 * family uses and there is never a second one.
 *
 * **The kit's `MatchupTable` is not used here.** It draws a rank and nothing
 * else, and a rating without its figure is half a fact. The kit's CSS and
 * `rankTier` are used, so this table is the matchup table with the number put
 * back next to the place.
 */
export function TeamRatings({
  ratings,
  awayTeam,
  homeTeam,
}: {
  ratings: GameRatings;
  awayTeam: string;
  homeTeam: string;
}) {
  const figure = (team: string, metric: RatingMetric) => ratings.teams?.[team]?.[metric.key];

  return (
    <section aria-labelledby="team-ratings-heading" data-testid="team-ratings">
      <h3
        id="team-ratings-heading"
        className="mb-1 font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-faint"
      >
        Team ratings vs the league
      </h3>
      <p className="mb-2 text-sm text-pr-text-dim">{ratings.caption}</p>
      <div className="pr-matchup-table-wrapper">
        <table className="pr-matchup-table">
          <caption className="sr-only" data-testid="team-ratings-caption">
            Ratings: {awayTeam} at {homeTeam}, ranked out of {ratings.n_teams} teams
          </caption>
          <thead>
            <tr>
              <th scope="col" className="pr-matchup-team">
                <span className="inline-flex items-center justify-center gap-1.5">
                  <TeamLogo team={awayTeam} size={20} ariaHidden />
                  {awayTeam}
                </span>
              </th>
              <th scope="col" className="pr-matchup-label-header">
                Stat
              </th>
              <th scope="col" className="pr-matchup-team">
                <span className="inline-flex items-center justify-center gap-1.5">
                  {homeTeam}
                  <TeamLogo team={homeTeam} size={20} ariaHidden />
                </span>
              </th>
            </tr>
          </thead>
          <tbody>
            {ratings.metrics.map((metric) => {
              const away = figure(awayTeam, metric);
              const home = figure(homeTeam, metric);
              return (
                <tr key={metric.key} data-testid={`rating-row-${metric.key}`}>
                  <td className="pr-matchup-rank">
                    <RatingValue figure={away} n={ratings.n_teams} label={metric.label} team={awayTeam} />
                  </td>
                  <td className="pr-matchup-label" title={metric.help}>
                    <RatingLabel metric={metric} />
                  </td>
                  <td className="pr-matchup-rank">
                    <RatingValue figure={home} n={ratings.n_teams} label={metric.label} team={homeTeam} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default TeamRatings;
