import { useEffect, useMemo, useRef, useState } from "react";
import { BIN_COLORS, BIN_NAMES, binColor, cellSize } from "../lib/display";

interface Props {
  grid: number[][];
  size?: number;
  /** Draw thin gaps between dies (off for thumbnails). */
  gaps?: boolean;
  interactive?: boolean;
  label?: string;
}

/**
 * Canvas die-grid renderer. Canvas (not one <div> per die) keeps a 300 x 205 map at
 * 60k dies cheap to draw, and the lot view renders 25 of them at once.
 */
export function WaferMap({ grid, size = 360, gaps = true, interactive = false, label }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{ r: number; c: number; bin: number } | null>(null);
  const rows = grid.length;
  const cols = grid[0]?.length ?? 0;
  const cell = cellSize(rows, cols, size);

  const counts = useMemo(() => {
    let pass = 0;
    let fail = 0;
    for (const row of grid) {
      for (const v of row) {
        if (v === 1) pass++;
        else if (v === 2) fail++;
      }
    }
    return { pass, fail };
  }, [grid]);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return; // jsdom has no 2-D context
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);
    const ox = (size - cols * cell) / 2;
    const oy = (size - rows * cell) / 2;
    const gap = gaps && cell >= 4 ? Math.max(0.5, cell * 0.08) : 0;
    for (let r = 0; r < rows; r++) {
      const row = grid[r]!;
      for (let c = 0; c < cols; c++) {
        const color = binColor(row[c]!);
        if (!color) continue;
        ctx.fillStyle = color;
        ctx.fillRect(ox + c * cell, oy + r * cell, cell - gap, cell - gap);
      }
    }
  }, [grid, size, rows, cols, cell, gaps]);

  function onMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const scale = size / rect.width;
    const x = (e.clientX - rect.left) * scale - (size - cols * cell) / 2;
    const y = (e.clientY - rect.top) * scale - (size - rows * cell) / 2;
    const r = Math.floor(y / cell);
    const c = Math.floor(x / cell);
    if (r >= 0 && r < rows && c >= 0 && c < cols) setHover({ r, c, bin: grid[r]![c]! });
    else setHover(null);
  }

  const summary = `${label ? `${label}: ` : ""}${rows} x ${cols} grid, ${counts.pass + counts.fail} dies, ${counts.fail} failing`;
  return (
    <figure className="wafer-map" style={{ width: size }}>
      <canvas
        ref={ref}
        role="img"
        aria-label={summary}
        style={{ width: size, height: size }}
        onMouseMove={interactive ? onMove : undefined}
        onMouseLeave={interactive ? () => setHover(null) : undefined}
      />
      {interactive && (
        <figcaption className="wafer-map__hover" aria-live="polite">
          {hover ? `row ${hover.r}, col ${hover.c}: ${BIN_NAMES[hover.bin]}` : "hover a die"}
        </figcaption>
      )}
    </figure>
  );
}

export function BinLegend() {
  return (
    <ul className="legend">
      <li>
        <span className="swatch" style={{ background: BIN_COLORS[1]! }} /> pass
      </li>
      <li>
        <span className="swatch" style={{ background: BIN_COLORS[2]! }} /> fail
      </li>
    </ul>
  );
}
