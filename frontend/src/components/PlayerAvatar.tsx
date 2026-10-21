import { useState } from "react";

/** ESPN's headshot for an NBA player, by athlete id. */
const HEADSHOT_BASE = "https://a.espncdn.com/i/headshots/nba/players/full";

/**
 * A player's headshot, with their initial when ESPN has no picture.
 *
 * The fallback is the same shape as `TeamLogo`'s: a circle with the initial
 * on it. A missing picture is not a broken image, and a row of broken images
 * is worse than a row of initials.
 */
export default function PlayerAvatar({
  playerId,
  name,
  size = 32,
  className = "",
}: {
  playerId: string;
  name: string;
  size?: number;
  className?: string;
}) {
  const [failed, setFailed] = useState(false);
  const initial = (name.trim().charAt(0) || "?").toUpperCase();

  if (!playerId || failed) {
    return (
      <span
        role="img"
        aria-label={`${name} (no picture)`}
        className={`inline-flex shrink-0 items-center justify-center rounded-full bg-[var(--color-net)] font-bold text-[var(--color-court-900)] ${className}`}
        style={{ width: size, height: size, fontSize: Math.round(size * 0.5) }}
      >
        {initial}
      </span>
    );
  }

  return (
    <img
      src={`${HEADSHOT_BASE}/${playerId}.png`}
      alt=""
      width={size}
      height={size}
      loading="lazy"
      onError={() => setFailed(true)}
      className={`shrink-0 rounded-full object-cover ${className}`}
    />
  );
}
