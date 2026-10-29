"""Write small **synthetic** wafer maps to ``frontend/public/samples/`` for the Classify page demo.

These are generated shapes (not WM-811K data), so they can be committed and served freely.
Each has light random background fails, like real maps.

    python scripts/make_sample_maps.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "frontend" / "public" / "samples"
SIZE = 44


def base(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = (np.arange(SIZE) + 0.5) - SIZE / 2
    yy, xx = np.meshgrid(c, c, indexing="ij")
    r = np.hypot(xx, yy) / (SIZE / 2)
    m = (r <= 1.0 - 0.5 / SIZE * 2).astype(np.uint8)
    m[(rng.random(m.shape) < 0.02) & (m > 0)] = 2  # background noise
    return m, r, np.arctan2(yy, xx)


def fail(m: np.ndarray, mask: np.ndarray, rng: np.random.Generator, p: float = 0.85) -> None:
    m[mask & (m > 0) & (rng.random(m.shape) < p)] = 2


def main() -> None:
    rng = np.random.default_rng(7)
    OUT.mkdir(parents=True, exist_ok=True)
    maps = {}

    m, r, _ = base(rng)
    fail(m, r < 0.28, rng)
    maps["center"] = m

    m, r, _ = base(rng)
    fail(m, r > 0.86, rng)
    maps["edge-ring"] = m

    m, r, _ = base(rng)
    fail(m, (r > 0.38) & (r < 0.6), rng)
    maps["donut"] = m

    m, r, _ = base(rng)
    rows = np.arange(8, 36)
    cols = np.clip((rows * 0.7 + 6).astype(int), 0, SIZE - 1)
    for dr, dc in ((0, 0), (0, 1)):
        sel = m[rows + dr, cols + dc] > 0
        m[(rows + dr)[sel], (cols + dc)[sel]] = 2
    maps["scratch"] = m

    m, _, _ = base(rng)
    maps["clean"] = m

    for name, grid in maps.items():
        (OUT / f"{name}.csv").write_text(
            "\n".join(",".join(map(str, row)) for row in grid) + "\n", newline="\n"
        )
        print(f"{name}: {(grid > 0).sum()} dies, {(grid == 2).sum()} fails")


if __name__ == "__main__":
    main()
