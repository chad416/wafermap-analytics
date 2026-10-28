// Pure display helpers (kept out of component files so React Fast Refresh works).

export const BIN_COLORS: Record<number, string | null> = {
  0: null, // no die: transparent
  1: "#c9d4e2", // pass
  2: "#d1343b", // fail
};
export const BIN_NAMES: Record<number, string> = { 0: "no die", 1: "pass", 2: "fail" };

export function binColor(bin: number): string | null {
  return BIN_COLORS[bin] ?? null;
}

/** Fit a rows x cols die grid into a `size` px square; returns the cell edge length in px. */
export function cellSize(rows: number, cols: number, size: number): number {
  return size / Math.max(rows, cols, 1);
}

// Okabe-Ito based categorical palette (colour-blind safe); 'none' is neutral grey.
export const PATTERN_COLORS: Record<string, string> = {
  Center: "#0072B2",
  Donut: "#56B4E9",
  "Edge-Loc": "#E69F00",
  "Edge-Ring": "#D55E00",
  Loc: "#009E73",
  "Near-full": "#CC79A7",
  Random: "#8B6F2F",
  Scratch: "#6A3D9A",
  none: "#9CA3AF",
};

export function fmtPct(v: number | null | undefined, digits = 1): string {
  return v === null || v === undefined ? "–" : `${v.toFixed(digits)} %`;
}
