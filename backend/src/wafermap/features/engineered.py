"""Hand-engineered wafer-map features for the gradient-boosting baseline.

Feature families (inspired by Wu et al., IEEE TSM 2015, adapted to a canonical grid):

* **Global** - fail ratio, die count, aspect ratio.
* **Density** - fail density in 5 radial rings, 8 angular sectors of the outer ring and a 3x3
  grid. Rings separate Center / Donut / Edge-Ring; sectors separate Edge-Loc from Edge-Ring.
* **Radon** - mean and std across projection angles of the Radon transform of the fail
  coverage, resampled to 20 points each. Line-shaped defects (Scratch) give a sharp peak at one
  angle, so the across-angle std is high.
* **Geometry** - shape of the largest connected fail cluster on the original die grid (area,
  perimeter, axes, eccentricity, solidity, distance from centre) and cluster statistics.

Spatial features are computed on the canonical 64x64 coverage grid so they do not depend on the
die size of the product; connected-component geometry uses the original grid because
resampling would merge or split clusters.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from skimage.measure import regionprops
from skimage.transform import radon

from wafermap.constants import BIN_FAIL, BIN_NO_DIE, CANONICAL_SIZE
from wafermap.wafer import WaferArray, canonical

N_RINGS = 5
N_SECTORS = 8
OUTER_RING_R = 0.7
RADON_ANGLES = np.arange(0.0, 180.0, 4.0)
RADON_POINTS = 20
_EIGHT = np.ones((3, 3), dtype=bool)


def feature_names() -> list[str]:
    names = ["fail_ratio", "log_n_die", "aspect_ratio"]
    names += [f"ring_{i}" for i in range(N_RINGS)]
    names += [f"sector_{i}" for i in range(N_SECTORS)]
    names += [f"grid_{r}{c}" for r in range(3) for c in range(3)]
    names += [f"radon_mean_{i}" for i in range(RADON_POINTS)]
    names += [f"radon_std_{i}" for i in range(RADON_POINTS)]
    names += [
        "n_clusters",
        "largest_cluster_frac_of_fails",
        "largest_area_frac_of_dies",
        "largest_perimeter_norm",
        "largest_major_axis_norm",
        "largest_minor_axis_norm",
        "largest_eccentricity",
        "largest_solidity",
        "largest_centroid_radius",
        "fail_centroid_radius",
        "fail_radius_std",
    ]
    return names


N_FEATURES = len(feature_names())


@lru_cache(maxsize=1)
def _polar_grid(size: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    c = (np.arange(size) + 0.5) / size * 2 - 1  # cell centres in [-1, 1]
    yy, xx = np.meshgrid(c, c, indexing="ij")
    return np.hypot(xx, yy), np.arctan2(yy, xx)


def _density(
    fail: npt.NDArray[np.float32], die: npt.NDArray[np.float32], mask: npt.NDArray[np.bool_]
) -> float:
    d = float(die[mask].sum())
    return float(fail[mask].sum()) / d if d > 1e-6 else 0.0


def _resample(profile: npt.NDArray[np.float64], n: int) -> npt.NDArray[np.float64]:
    x_old = np.linspace(0.0, 1.0, profile.size)
    return np.interp(np.linspace(0.0, 1.0, n), x_old, profile)


def extract(m: WaferArray) -> npt.NDArray[np.float32]:
    """Return the feature vector (length :data:`N_FEATURES`) for one wafer map."""
    die_c, fail_c = canonical(m)
    size = CANONICAL_SIZE
    r, theta = _polar_grid(size)
    # Normalise the radius by the wafer's actual extent so partial wafers are handled.
    on_wafer = die_c > 0.5
    r_max = float(r[on_wafer].max()) if on_wafer.any() else 1.0
    rn = r / max(r_max, 1e-6)

    die_mask = m != BIN_NO_DIE
    fail_mask = m == BIN_FAIL
    n_die = int(die_mask.sum())
    n_fail = int(fail_mask.sum())
    feats: list[float] = [n_fail / n_die, float(np.log(n_die)), m.shape[1] / m.shape[0]]

    edges = np.linspace(0.0, 1.0 + 1e-9, N_RINGS + 1)
    edges[-1] = np.inf
    feats += [
        _density(fail_c, die_c, (rn >= edges[i]) & (rn < edges[i + 1])) for i in range(N_RINGS)
    ]

    sector = ((theta + np.pi) / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS
    outer = rn >= OUTER_RING_R
    feats += [_density(fail_c, die_c, outer & (sector == s)) for s in range(N_SECTORS)]

    third = size // 3
    bounds = [0, third, 2 * third, size]
    for i in range(3):
        for j in range(3):
            block = np.zeros((size, size), dtype=bool)
            block[bounds[i] : bounds[i + 1], bounds[j] : bounds[j + 1]] = True
            feats.append(_density(fail_c, die_c, block))

    sino = radon(fail_c.astype(np.float64), theta=RADON_ANGLES, circle=False)
    sino /= size  # scale-free: projection sums -> mean coverage along the ray
    feats += _resample(sino.mean(axis=1), RADON_POINTS).tolist()
    feats += _resample(sino.std(axis=1), RADON_POINTS).tolist()

    feats += _geometry(m, die_mask, fail_mask, n_die, n_fail)
    return np.asarray(feats, dtype=np.float32)


def _geometry(
    m: WaferArray,
    die_mask: npt.NDArray[np.bool_],
    fail_mask: npt.NDArray[np.bool_],
    n_die: int,
    n_fail: int,
) -> list[float]:
    rows, cols = np.nonzero(die_mask)
    cy, cx = rows.mean(), cols.mean()
    radius = max(np.hypot(rows - cy, cols - cx).max(), 1.0)
    diameter = 2 * radius
    if n_fail == 0:
        return [0.0] * 11
    fr, fc = np.nonzero(fail_mask)
    fail_r = np.hypot(fr - cy, fc - cx) / radius
    labels, n_clusters = ndimage.label(fail_mask, structure=_EIGHT)
    sizes = np.bincount(labels.ravel())[1:]
    largest = int(np.argmax(sizes)) + 1
    props = regionprops((labels == largest).astype(np.uint8))[0]
    pr, pc = props.centroid
    return [
        float(n_clusters),
        float(sizes.max()) / n_fail,
        float(props.area) / n_die,
        float(props.perimeter) / np.sqrt(n_die),
        float(props.axis_major_length) / diameter,
        float(props.axis_minor_length) / diameter,
        float(props.eccentricity),
        float(props.solidity),
        float(np.hypot(pr - cy, pc - cx) / radius),
        float(fail_r.mean()),
        float(fail_r.std()),
    ]
