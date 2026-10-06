import { useState } from "react";
import { teamLogoUrl } from "../data/teamLogos";

interface TeamLogoProps {
  team: string;
  size?: number;
  className?: string;
}

export default function TeamLogo({ team, size = 20, className = "" }: TeamLogoProps) {
  const [failedUrl, setFailedUrl] = useState<string | null>(null);
  const url = teamLogoUrl(team);
  const initial = (team.trim().charAt(0) || "?").toUpperCase();

  // Keyed by URL, not a bare boolean. A plain `failed` flag was sticky: once a
  // logo errored, this instance showed the fallback for every *later* team
  // without ever attempting the new URL -- so a card showing both teams would
  // put the wrong team's initial on the half that was fine. Storing which URL
  // failed also drops the fallback automatically when the team changes to one
  // that has not failed, with no effect needed at all for that case.
  const failed = failedUrl !== null && failedUrl === url;

  if (!url || failed) {
    return (
      <span
        role="img"
        aria-label={`${team} logo`}
        className={`inline-flex shrink-0 items-center justify-center rounded-full bg-[var(--color-net)] font-bold text-[var(--color-court-900)] ${className}`}
        style={{ width: size, height: size, fontSize: Math.round(size * 0.55) }}
      >
        {initial}
      </span>
    );
  }

  return (
    <img
      src={url}
      alt={`${team} logo`}
      width={size}
      height={size}
      loading="lazy"
      onError={() => setFailedUrl(url)}
      className={`shrink-0 object-contain ${className}`}
    />
  );
}
