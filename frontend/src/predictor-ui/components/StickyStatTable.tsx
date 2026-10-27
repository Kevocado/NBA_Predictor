// Synced from predictor-ui@a141469b1a04. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { useState, type ReactNode } from "react";

/**
 * The phone-readable stat table: players down the side, stats across the top,
 * the stats slide sideways inside their own scroller and the first column
 * (the full player name) never moves. Presentational: it renders rows the
 * caller has built and decides nothing about order or content.
 */

export type StickyColumn = { key: string; label: string };

export type StickyRow = {
  key: string;
  /** First, pinned cell: the full name, plus whatever marker the data has. */
  header: ReactNode;
  /** Aligned to `columns`. null / undefined / NaN / "" render an em dash. */
  cells: ReactNode[];
  /** Raw values used for sorting, aligned to `columns`. Falls back to the
   *  first column's header text when absent. */
  sortValues?: (number | string | null)[];
  /** Screen-reader text per cell, aligned to `cells`. */
  cellLabels?: (string | undefined)[];
  /** Index of the one cell that shouts. */
  emphasisIndex?: number;
  /** A total / team row: bolder, ruled off, and (in `footer`) last. */
  total?: boolean;
  testId?: string;
  attrs?: Record<string, string>;
};

export type StickySection = { label?: string; rows: StickyRow[] };

export interface StickyStatTableProps {
  columns: StickyColumn[];
  sections: StickySection[];
  /** Rows after every section (team totals), rendered in <tfoot>. */
  footer?: StickyRow[];
  /** Visually hidden table caption. */
  caption: string;
  firstColumnLabel?: string;
  ariaLabel?: string;
  /** One-line message shown instead of an empty table. */
  empty?: string;
  testId?: string;
  tableTestId?: string;
}

const EM_DASH = "—";
/** Pinned-column width: full names wrap to two lines inside it rather than clip. */
const NAME_WIDTH = "10rem";

function show(cell: ReactNode): ReactNode {
  if (cell === null || cell === undefined || cell === "") return EM_DASH;
  if (typeof cell === "number") {
    if (!Number.isFinite(cell)) return EM_DASH;
    return Number.isInteger(cell) ? String(cell) : cell.toFixed(1);
  }
  return cell;
}

// Opaque on purpose: a translucent pinned cell lets scrolled stats show through.
const bgFor = (i: number) => (i % 2 === 1 ? "bg-pr-panel-2" : "bg-pr-panel");

function sortValueOf(row: StickyRow, index: number): number | string | null {
  if (row.sortValues && index < row.sortValues.length) return row.sortValues[index] ?? null;
  return null;
}

function compareSticky(a: number | string | null, b: number | string | null, dir: "asc" | "desc"): number {
  if (a === null && b === null) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  // Dashes / DNP sort last in both directions.
  const aStr = typeof a === "string" ? a.trim().toLowerCase() : null;
  const bStr = typeof b === "string" ? b.trim().toLowerCase() : null;
  if (aStr === "—" || aStr === "dnp" || aStr === "-" || aStr === "") return 1;
  if (bStr === "—" || bStr === "dnp" || bStr === "-" || bStr === "") return -1;
  let an: number | null = null;
  let bn: number | null = null;
  if (typeof a === "number") an = a;
  else if (typeof a === "string") {
    const t = a.trim();
    if (/^\d+:\d{2}(:\d{2})?$/.test(t)) {
      const parts = t.split(":").map(Number);
      an = parts.reduce((acc, p) => acc * 60 + p, 0);
    } else if (t.endsWith("%")) {
      const n = Number(t.slice(0, -1));
      if (!Number.isNaN(n)) an = n;
    } else {
      const n = Number(t);
      if (!Number.isNaN(n) && t !== "") an = n;
    }
  }
  if (typeof b === "number") bn = b;
  else if (typeof b === "string") {
    const t = b.trim();
    if (/^\d+:\d{2}(:\d{2})?$/.test(t)) {
      const parts = t.split(":").map(Number);
      bn = parts.reduce((acc, p) => acc * 60 + p, 0);
    } else if (t.endsWith("%")) {
      const n = Number(t.slice(0, -1));
      if (!Number.isNaN(n)) bn = n;
    } else {
      const n = Number(t);
      if (!Number.isNaN(n) && t !== "") bn = n;
    }
  }
  let order: number;
  if (an !== null && bn !== null) order = an - bn;
  else order = String(a).localeCompare(String(b));
  return dir === "asc" ? order : -order;
}

export function StickyStatTable({
  columns,
  sections,
  footer = [],
  caption,
  firstColumnLabel = "Player",
  ariaLabel = "Box score, scroll sideways for more stats",
  empty,
  testId,
  tableTestId,
}: StickyStatTableProps) {
  const [pinned, setPinned] = useState(false);
  const [sort, setSort] = useState<{ index: number; dir: "asc" | "desc" } | null>(null);
  const rowCount = sections.reduce((n, s) => n + s.rows.length, 0) + footer.length;

  if (rowCount === 0) {
    return empty ? (
      <p className="py-3 text-sm text-pr-text-dim" data-testid={testId}>
        {empty}
      </p>
    ) : null;
  }

  const pin = {
    boxShadow: pinned ? "4px 0 6px -2px rgba(0,0,0,0.5)" : undefined,
    borderRight: "1px solid var(--color-pr-rule)",
  };
  const nameStyle = { width: NAME_WIDTH, minWidth: NAME_WIDTH, maxWidth: NAME_WIDTH, ...pin };
  const nameCls = "sticky left-0 z-10 whitespace-normal break-words px-3 py-1.5 text-left align-middle text-[0.8125rem] font-normal leading-tight text-pr-text";

  const toggleSort = (index: number) => {
    setSort((cur) => {
      if (!cur || cur.index !== index) return { index, dir: "desc" };
      if (cur.dir === "desc") return { index, dir: "asc" };
      return null;
    });
  };

  const sortedSections = sort
    ? sections.map((section) => ({
        ...section,
        rows: [...section.rows].sort((a, b) => compareSticky(sortValueOf(a, sort.index), sortValueOf(b, sort.index), sort.dir)),
      }))
    : sections;

  const renderRow = (row: StickyRow, i: number) => {
    const bg = row.total ? "bg-pr-panel-2" : bgFor(i);
    return (
      <tr
        key={row.key}
        data-testid={row.testId}
        {...row.attrs}
        className={`border-t border-pr-rule ${row.total ? "font-semibold" : ""} ${bg}`}
      >
        <th scope="row" data-pinned={String(pinned)} className={`${nameCls} ${bg} ${row.total ? "font-semibold" : ""}`} style={nameStyle}>
          {row.header}
        </th>
        {columns.map((column, c) => {
          const label = row.cellLabels?.[c];
          return (
            <td
              key={column.key}
              {...(label ? { "aria-label": label } : {})}
              className={`${bg} whitespace-nowrap px-3 py-1.5 text-right align-middle text-[0.8125rem] tabular-nums ${row.emphasisIndex === c || row.total ? "font-semibold" : ""}`}
            >
              {show(row.cells[c])}
            </td>
          );
        })}
      </tr>
    );
  };

  const headerButton = (label: string, index: number, alignRight: boolean) => {
    const active = sort?.index === index;
    return (
      <button
        type="button"
        onClick={() => toggleSort(index)}
        className={`inline-flex items-center gap-1 uppercase tracking-wide hover:text-pr-text ${active ? "text-pr-text" : ""}`}
      >
        {label}
        <span aria-hidden="true">{active ? (sort!.dir === "asc" ? "↑" : "↓") : ""}</span>
      </button>
    );
  };

  return (
    <div data-testid={testId}>
      <div
        role="region"
        tabIndex={0}
        aria-label={ariaLabel}
        onScroll={(e) => setPinned(e.currentTarget.scrollLeft > 0)}
        className="max-h-[min(75vh,44rem)] overflow-auto rounded-pr border border-pr-rule [-webkit-overflow-scrolling:touch] focus-visible:outline focus-visible:outline-2 focus-visible:outline-pr-accent"
      >
        <table className="w-max min-w-full border-collapse" data-testid={tableTestId}>
          <caption className="sr-only">{caption}</caption>
          <thead>
            <tr>
              <th
                scope="col"
                data-pinned={String(pinned)}
                aria-sort={sort?.index === 0 ? (sort!.dir === "asc" ? "ascending" : "descending") : "none"}
                className="sticky left-0 top-0 z-30 bg-pr-panel px-3 py-1.5 text-left text-[0.6875rem] font-semibold uppercase tracking-[0.04em] text-pr-text-dim"
                style={nameStyle}
              >
                {headerButton(firstColumnLabel, 0, false)}
              </th>
              {columns.map((column, c) => {
                const idx = c + 1;
                return (
                  <th
                    key={column.key}
                    scope="col"
                    aria-sort={sort?.index === idx ? (sort!.dir === "asc" ? "ascending" : "descending") : "none"}
                    className="sticky top-0 z-20 whitespace-nowrap bg-pr-panel px-3 py-1.5 text-right text-[0.6875rem] font-semibold uppercase tracking-[0.04em] text-pr-text-dim"
                  >
                    {headerButton(column.label, idx, true)}
                  </th>
                );
              })}
            </tr>
          </thead>
          {sortedSections.map((section, s) => (
            <tbody key={section.label ?? s}>
              {section.label && (
                <tr className="border-t border-pr-rule bg-pr-panel-2">
                  <th
                    scope="rowgroup"
                    data-pinned={String(pinned)}
                    className="sticky left-0 z-10 bg-pr-panel-2 px-3 py-1 text-left font-pr-display text-xs font-bold uppercase tracking-wider text-pr-text"
                    style={nameStyle}
                  >
                    {section.label}
                  </th>
                  <td colSpan={columns.length} className="bg-pr-panel-2" />
                </tr>
              )}
              {section.rows.map(renderRow)}
            </tbody>
          ))}
          {footer.length > 0 && <tfoot>{footer.map(renderRow)}</tfoot>}
        </table>
      </div>
    </div>
  );
}
