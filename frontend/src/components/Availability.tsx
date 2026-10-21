import type { DoubtfulPlayer, OutPlayer } from "../api/client";

/**
 * Who is not playing, for the two teams in this game.
 *
 * The out list is the report's own: a player who is Out leaves the ranking,
 * and a player who is day-to-day stays in it but is said out loud. Both are
 * named here with the team they play for, because a reader who sees "Out"
 * without a team cannot tell whether it affects this game.
 *
 * **This game's two teams only.** ESPN's report covers the whole league, and
 * a player playing for neither side is not this game's news.
 */
export function Availability({
  out,
  doubtful,
}: {
  out: OutPlayer[];
  doubtful: DoubtfulPlayer[];
}) {
  if (out.length === 0 && doubtful.length === 0) return null;

  return (
    <section aria-labelledby="availability-heading" data-testid="availability">
      <h3
        id="availability-heading"
        className="mb-1 font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-faint"
      >
        Availability
      </h3>
      <ul className="flex flex-col gap-1 text-sm">
        {out.map((player) => (
          <li key={`out-${player.player_id}`} data-testid="availability-out" className="flex items-baseline gap-2">
            <span className="font-semibold text-pr-text">{player.player_name}</span>
            <span className="text-pr-text-dim">{player.team}</span>
            <span className="text-pr-loss">Out</span>
          </li>
        ))}
        {doubtful.map((player) => (
          <li key={`doubtful-${player.player_id}`} data-testid="availability-doubtful" className="flex items-baseline gap-2">
            <span className="font-semibold text-pr-text">{player.player_name}</span>
            <span className="text-pr-text-dim">{player.team}</span>
            <span className="text-pr-text-dim">Day-to-day</span>
          </li>
        ))}
      </ul>
      <p className="mt-1 text-xs text-pr-text-faint">
        From {out[0]?.source ?? doubtful[0]?.source ?? "ESPN's injury report"}
        {out[0]?.dated ? `, ${out[0].dated.slice(0, 10)}` : ""}
      </p>
    </section>
  );
}

export default Availability;
