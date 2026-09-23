import { useState } from "react";
import { teamLogoUrl } from "../data/teamLogos";

interface TeamLogoProps {
  team: string;
  size?: number;
  className?: string;
}

export default function TeamLogo({ team, size = 20, className = "" }: TeamLogoProps) {
  const [failed, setFailed] = useState(false);
  const url = teamLogoUrl(team);
  const initial = (team.trim().charAt(0) || "?").toUpperCase();

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
      onError={() => setFailed(true)}
      className={`shrink-0 object-contain ${className}`}
    />
  );
}
