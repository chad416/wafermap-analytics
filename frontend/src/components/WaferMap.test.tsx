import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { binColor, cellSize } from "../lib/display";
import { WaferMap } from "./WaferMap";

const grid = [
  [0, 1, 1, 0],
  [1, 2, 2, 1],
  [1, 1, 2, 1],
  [0, 1, 1, 0],
];

describe("WaferMap", () => {
  it("maps bins to colours with no-die transparent", () => {
    expect(binColor(0)).toBeNull();
    expect(binColor(1)).not.toBe(binColor(2));
    expect(binColor(7)).toBeNull();
  });

  it("fits the longer side of the grid into the canvas", () => {
    expect(cellSize(25, 50, 200)).toBe(4);
    expect(cellSize(0, 0, 100)).toBe(100);
  });

  it("describes the map for screen readers", () => {
    render(<WaferMap grid={grid} size={120} label="w1" />);
    const img = screen.getByRole("img");
    expect(img).toHaveAttribute("aria-label", "w1: 4 x 4 grid, 12 dies, 3 failing");
    expect(img).toHaveStyle({ width: "120px" });
  });
});
