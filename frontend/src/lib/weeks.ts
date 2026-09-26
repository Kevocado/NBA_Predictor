// Week arithmetic on plain "YYYY-MM-DD" dates, done in UTC so the result is
// the same calendar date in every time zone (local midnight read back through
// toISOString() slipped a day east of UTC, landing weeks on Sunday).

const DAY_MS = 86_400_000;

export function addDays(isoDate: string, days: number): string {
  return new Date(Date.parse(`${isoDate}T00:00:00Z`) + days * DAY_MS).toISOString().slice(0, 10);
}

/** Monday of the viewer's own calendar day. */
export function mondayOf(d: Date): string {
  const today = Date.UTC(d.getFullYear(), d.getMonth(), d.getDate());
  const sinceMonday = (d.getDay() + 6) % 7; // Monday 0 … Sunday 6
  return new Date(today - sinceMonday * DAY_MS).toISOString().slice(0, 10);
}
