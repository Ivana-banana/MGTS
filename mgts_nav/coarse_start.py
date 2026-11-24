"""Experimental, bounded coarse-to-fine single-scan search on a fixed map."""
from time import perf_counter

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from .initialization import _same_pose
from .slam import match, rotation, world


MAX_TILES = 2400
MAX_BASINS = 32
TIME_LIMIT_S = 8.0


def _sample(points, limit):
    return points[np.linspace(0, len(points)-1, min(len(points), limit), dtype=int)]


def _scores(field, seeds, offsets):
    """Mean endpoint likelihood; points outside the grid contribute zero."""
    h, w = field.shape
    x, y = (seeds[:, None, :] + offsets).transpose(2, 0, 1)
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    return (field[y.clip(0, h-1), x.clip(0, w-1)] * valid).mean(axis=1)


def estimate(scan, pixels, meta, stats):
    """Return pose in map coordinates or raise with partial search diagnostics."""
    deadline = perf_counter() + TIME_LIMIT_S
    stats.update(coarse=0, fine=0, icp=0)
    def check_time():
        if perf_counter() >= deadline:
            raise ValueError("Search time budget exhausted")

    cell = float(meta["resolution"])
    origin = np.asarray(meta["origin"], dtype=float)
    if (pixels.ndim != 2 or pixels.size > 4_000_000 or not np.isfinite(cell) or cell <= 0
            or origin.shape != (3,) or not np.isfinite(origin).all() or origin[2] or meta.get("negate", 0)):
        raise ValueError("Expected bounded, axis-aligned non-negated map")
    xy = np.asarray(scan.xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) > 100_000:
        raise ValueError("Expected up to 100,000 planar endpoints")
    if np.shape(scan.origin) != (2,) or not np.isfinite(scan.origin).all():
        raise ValueError("Invalid lidar origin")
    # Grid.observe limits mapping rays to 35 m from the sensor origin.
    xy = xy[np.isfinite(xy).all(axis=1) & (np.linalg.norm(xy-scan.origin, axis=1) <= 35)]
    xy = np.unique(xy, axis=0)
    if len(xy) < 12:
        raise ValueError("Too few in-range endpoints")
    xy = xy[np.argsort(np.arctan2(xy[:, 1]-scan.origin[1], xy[:, 0]-scan.origin[0]), kind="stable")]
    occupancy = 1 - np.flipud(pixels).astype(np.float32)/255
    occupied = occupancy > meta["occupied_thresh"]
    free = occupancy < meta["free_thresh"]
    cells = np.argwhere(free)[:, ::-1]
    if len(cells) == 0 or occupied.sum() < 12:
        raise ValueError("Insufficient known map geometry")
    stride = max(1, round(.6 / cell))
    while True:
        check_time()
        _, first = np.unique(cells // stride, axis=0, return_index=True)
        if len(first) <= MAX_TILES:
            break
        stride = max(stride+1, int(np.ceil(stride*np.sqrt(len(first)/MAX_TILES))))
    seeds = cells[first]
    stats.update(tiles=len(seeds), tile_m=stride*cell, endpoints=len(xy))
    distance = distance_transform_edt(~occupied)*cell
    broad = np.exp(-.5*(distance/max(.5, stride*cell))**2).astype(np.float32)
    sharp = np.exp(-.5*(distance/.18)**2).astype(np.float32)
    sample = _sample(xy, 96)
    proposals = []
    for deg in range(-180, 180, 10):
        check_time()
        yaw = np.deg2rad(deg)
        offsets = np.rint(sample @ rotation(yaw).T / cell).astype(int)
        scores = _scores(broad, seeds, offsets)
        stats["coarse"] += len(seeds)
        # Keep separated translations for every orientation, even on repetitions.
        for _ in range(min(6, len(seeds))):
            i = scores.argmax()
            if scores[i] <= 0:
                break
            proposals.append((scores[i], seeds[i], yaw))
            scores[np.linalg.norm((seeds-seeds[i])*cell, axis=1) < max(.8, stride*cell)] = -1

    sample = _sample(xy, 224)
    step = np.unique(np.rint(np.linspace(-stride, stride, 7)).astype(int))
    shifts = np.stack(np.meshgrid(step, step), axis=-1).reshape(-1, 2)
    basins, fine = [], []
    h, w = free.shape
    for _, seed, yaw in sorted(proposals, key=lambda item: -item[0]):
        check_time()
        rough = np.r_[origin[:2]+(seed+.5)*cell, yaw]
        if any(_same_pose(rough, p, .6, np.deg2rad(9)) for p in basins):
            continue
        basins.append(rough)
        loc = seed + shifts
        inside = (loc[:, 0]>=0) & (loc[:, 0]<w) & (loc[:, 1]>=0) & (loc[:, 1]<h)
        loc = loc[inside]
        loc = loc[free[loc[:, 1], loc[:, 0]]]
        best_score = -1
        for angle in yaw + np.deg2rad([-5, -2.5, 0, 2.5, 5]):
            offsets = np.rint(sample @ rotation(angle).T/cell).astype(int)
            scores = _scores(sharp, loc, offsets)
            stats["fine"] += len(loc)
            i = scores.argmax()
            if scores[i] > best_score:
                best_score = scores[i]
                best = np.r_[origin[:2]+(loc[i]+.5)*cell, angle]
        fine.append((best_score, best))
        if len(basins) == MAX_BASINS:
            break

    tree = cKDTree(origin[:2] + (np.argwhere(occupied)[:, ::-1]+.5)*cell)
    modes = []
    for _, seed in sorted(fine, key=lambda item: -item[0]):
        check_time()
        stats["icp"] += 1
        p, _, ok = match(_sample(xy, 768), tree, seed, radius=.65)
        if not ok:
            continue
        x, y = np.floor((p[:2]-origin[:2])/cell).astype(int)
        if not (0 <= x < w and 0 <= y < h and free[y, x]):
            continue
        d = tree.query(world(xy, p))[0]
        modes.append((float(np.exp(-.5*(d/max(.10, cell))**2).mean()), p))
    check_time()
    modes.sort(key=lambda item: -item[0])
    if not modes:
        raise ValueError("No supported refinement")
    score, best = modes[0]
    stats["best_score"] = score
    stats["best_refined_pose"] = best.tolist()
    rival = next((s for s, p in modes if not _same_pose(p, best, .45, np.deg2rad(12))), None)
    stats["rival_score"] = rival
    if score < .55:
        raise ValueError("Weak final support")
    if rival is not None and score-rival <= .045:
        raise ValueError("Ambiguous global pose")
    best[2] = np.arctan2(np.sin(best[2]), np.cos(best[2]))
    return best
