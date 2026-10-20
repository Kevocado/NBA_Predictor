/** About ten games per team across the league's 30 teams (30 * 10 / 2). */
export const EARLY_SEASON_GAMES = 150;

/**
 * The line shown on the Games page while the model is young. `seen` is how many of this
 * season's games the served model was trained on (the manifest says so); unknown means
 * no claim, because guessing here would be the opposite of the notice's job.
 */
export function earlySeasonNotice(seen: number | null | undefined): string | null {
  if (seen == null || !Number.isFinite(seen) || seen >= EARLY_SEASON_GAMES) return null;
  const learned = seen === 0 ? "has not seen a game of this season yet" : `has learned from ${seen} of this season's games so far`;
  return `Season just started. The model ${learned}, so early picks lean on last season and are less reliable. It settles in after about ten games per team.`;
}
