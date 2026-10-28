import type { ReactNode } from "react";
import { PATTERNS } from "../api/types";
import { PATTERN_COLORS } from "../lib/display";

export function PatternBadge({ label, muted = false }: { label: string | null; muted?: boolean }) {
  if (!label) return <span className="badge badge--empty">unlabelled</span>;
  return (
    <span
      className={`badge${muted ? " badge--muted" : ""}`}
      style={{ borderColor: PATTERN_COLORS[label] }}
    >
      <span className="dot" style={{ background: PATTERN_COLORS[label] }} />
      {label}
    </span>
  );
}

export function PatternSelect({
  value,
  onChange,
  allLabel = "All patterns",
  id,
}: {
  value: string;
  onChange: (v: string) => void;
  allLabel?: string;
  id?: string;
}) {
  return (
    <select id={id} value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">{allLabel}</option>
      {PATTERNS.map((p) => (
        <option key={p} value={p}>
          {p}
        </option>
      ))}
    </select>
  );
}

export function StatCard({
  label,
  value,
  hint,
}: {
  label: string;
  value: ReactNode;
  hint?: string;
}) {
  return (
    <div className="stat">
      <div className="stat__label">{label}</div>
      <div className="stat__value">{value}</div>
      {hint && <div className="stat__hint">{hint}</div>}
    </div>
  );
}

export function Pagination({
  page,
  pages,
  total,
  onPage,
}: {
  page: number;
  pages: number;
  total: number;
  onPage: (p: number) => void;
}) {
  return (
    <nav className="pagination" aria-label="pagination">
      <button disabled={page <= 1} onClick={() => onPage(page - 1)}>
        Previous
      </button>
      <span>
        Page {page} of {pages} · {total.toLocaleString()} total
      </span>
      <button disabled={page >= pages} onClick={() => onPage(page + 1)}>
        Next
      </button>
    </nav>
  );
}

export function Loading() {
  return <div className="state">Loading…</div>;
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="state state--error" role="alert">
      {message}
    </div>
  );
}
