"""Bounded global initialization from an odometry-registered LiDAR prefix."""
from itertools import islice

import numpy as np

from .initialization import estimate_initial
from .slam import relative, world


MAX_SCANS = 48
WINDOW_S = 3.0
MAX_VIEWS = 8
MAX_POINTS_PER_VIEW = 256
MAX_SEARCH_TILES = 5000


def _registered_prefix(scans):
    eligible = []
    for scan in islice(scans, MAX_SCANS):
        if not np.isfinite(scan.t) or (eligible and scan.t <= eligible[-1].t):
            raise ValueError("Multi-scan initialization needs strictly increasing timestamps")
        if eligible and scan.t - eligible[0].t > WINDOW_S:
            break
        if np.shape(scan.odom) != (3,) or not np.isfinite(scan.odom).all():
            raise ValueError("Missing odometry within initialization window")
        eligible.append(scan)
    if len(eligible) < 2:
        raise ValueError("At least two scan timestamps are required")

    indices = np.linspace(0, len(eligible) - 1, min(len(eligible), MAX_VIEWS), dtype=int)
    clouds = []
    for i in indices:
        scan = eligible[i]
        xy = np.asarray(scan.xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("Expected N x 2 LiDAR endpoints")
        xy = xy[np.isfinite(xy).all(axis=1)]
        if not len(xy):
            continue
        sample = np.linspace(0, len(xy) - 1, min(len(xy), MAX_POINTS_PER_VIEW), dtype=int)
        clouds.append(world(xy[sample], relative(eligible[0].odom, scan.odom)))
    if len(clouds) < 2:
        raise ValueError("At least two non-empty views are required")
    points = np.vstack(clouds)
    _, indices = np.unique(np.floor(points / 0.06), axis=0, return_index=True)
    return points[indices]


def estimate_multiscan(scans, pixels, meta):
    """Return pose at the first scan in map coordinates, or raise ValueError.

    Reads at most the first 48 observations and accepts up to 3 s of them.
    No initial map pose or external trajectory is needed; only local odometry
    and map geometry participate. The global search retains its ambiguity gate.
    """
    cell = float(meta["resolution"])
    shape = np.shape(pixels)
    if len(shape) != 2 or not np.isfinite(cell) or cell <= 0:
        raise ValueError("Invalid map resolution or pixel dimensions")
    stride = max(1, round(0.30 / cell))
    tile_count = ((shape[0] + stride - 1) // stride) * ((shape[1] + stride - 1) // stride)
    if tile_count > MAX_SEARCH_TILES or shape[0] * shape[1] > 4_000_000:
        raise ValueError("Map exceeds the bounded global search budget")
    return estimate_initial(_registered_prefix(scans), pixels, meta)
