"""Single-scan global localization on an axis-aligned occupancy grid."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .slam import match, rotation, world


def _same_pose(a, b, distance, angle):
    dyaw = a[2] - b[2]
    return (np.linalg.norm(a[:2] - b[:2]) < distance and
            abs(np.arctan2(np.sin(dyaw), np.cos(dyaw))) < angle)


def estimate_initial(xy, pixels, meta):
    """Return [x, y, yaw] in map coordinates, or raise on weak/ambiguous fits."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError('Expected lidar endpoints with shape (N, 2)')
    xy = np.unique(xy[np.isfinite(xy).all(axis=1)], axis=0)
    if len(xy) < 12:
        raise ValueError('Too few scan endpoints for global localization')
    cell = float(meta['resolution'])
    origin = np.asarray(meta['origin'], dtype=float)
    if cell <= 0 or not np.isfinite(cell) or origin.shape != (3,) or not np.isfinite(origin).all() or origin[2] or meta.get('negate', 0):
        raise ValueError('Expected a finite axis-aligned non-negated map')
    occupancy = 1 - np.flipud(np.asarray(pixels, dtype=float)) / 255
    occupied = occupancy > meta['occupied_thresh']
    free = occupancy < meta['free_thresh']
    if occupied.sum() < 12 or not free.any():
        raise ValueError('Insufficient known map geometry')
    h, w = free.shape
    tree = cKDTree(origin[:2] + (np.argwhere(occupied)[:, ::-1] + .5) * cell)
    stride = max(1, round(.30 / cell))
    # Pick a real free cell in every coarse tile, including narrow passages.
    cells = np.argwhere(free)[:, ::-1]
    _, indices = np.unique(cells // stride, axis=0, return_index=True)
    seeds = cells[indices]
    field = np.exp(-.5 * (distance_transform_edt(~occupied) * cell / .30) ** 2)
    sample = xy[np.linspace(0, len(xy)-1, min(len(xy), 96), dtype=int)]
    candidates = []
    for yaw in np.deg2rad(np.arange(-180, 180, 5)):
        offsets = np.rint(sample @ rotation(yaw).T / cell).astype(int)
        scores = np.empty(len(seeds))
        for start in range(0, len(seeds), 512):
            batch = seeds[start:start+512]
            ix, iy = (batch[:, None] + offsets).transpose(2, 0, 1)
            valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            scores[start:start+len(batch)] = (
                field[iy.clip(0, h-1), ix.clip(0, w-1)] * valid).mean(axis=1)
        for _ in range(min(6, len(seeds))):
            best = scores.argmax()
            if scores[best] <= 0:
                break
            seed = seeds[best]
            candidates.append((scores[best], np.r_[origin[:2] + (seed+.5)*cell, yaw]))
            scores[np.sum(((seeds - seed)*cell)**2, axis=1) < .55**2] = -1

    refined = []
    tried = []
    for _, seed in sorted(candidates, key=lambda item: -item[0]):
        if any(_same_pose(seed, old, .55, np.deg2rad(12)) for old in tried):
            continue
        tried.append(seed)
        pose, _, ok = match(xy, tree, seed, radius=.65)
        if ok:
            ix, iy = np.floor((pose[:2] - origin[:2]) / cell).astype(int)
            if 0 <= ix < w and 0 <= iy < h and free[iy, ix]:
                distances = tree.query(world(xy, pose))[0]
                score = np.exp(-.5 * (distances / max(.10, cell))**2).mean()
                refined.append((score, pose))
        if len(tried) >= 48:
            break
    # Trimmed ICP can stop one cell away along a straight wall. Rescore each
    # distinct basin after optimizing the same likelihood used for comparison.
    def loss(pose):
        distances = tree.query(world(xy, pose))[0]
        return -np.exp(-.5 * (distances / max(.10, cell))**2).mean()

    modes = []
    for _, pose in sorted(refined, key=lambda item: -item[0]):
        if any(_same_pose(pose, old, .35, np.deg2rad(10)) for _, old in modes):
            continue
        bounds = np.column_stack((pose - [.15, .15, .09], pose + [.15, .15, .09]))
        fit = minimize(loss, pose, method='Powell', bounds=bounds,
                       options={'maxiter': 12, 'xtol': .0005})
        ix, iy = np.floor((fit.x[:2] - origin[:2]) / cell).astype(int)
        if 0 <= ix < w and 0 <= iy < h and free[iy, ix] and fit.fun <= loss(pose):
            pose = fit.x
        modes.append((-loss(pose), pose))
    refined = modes
    if not refined or max(score for score, _ in refined) < .55:
        raise ValueError('No sufficiently supported global pose')
    refined.sort(key=lambda item: -item[0])
    best_score, best_pose = refined[0]
    for score, pose in refined[1:]:
        if best_score - score <= .045 and not _same_pose(pose, best_pose, .45, np.deg2rad(12)):
            raise ValueError('Ambiguous global pose: multiple scan-to-map matches')
    best_pose[2] = np.arctan2(np.sin(best_pose[2]), np.cos(best_pose[2]))
    return best_pose
