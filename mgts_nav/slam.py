"""Planar scan matching in map coordinates, with body-frame motion increments."""
from collections import deque
from dataclasses import dataclass
import json

import numpy as np
from scipy.spatial import cKDTree


MIN_MATCH_POINTS = 12
# Wall-surface targets for localization, see WallSurface.
SURFACE_RADIUS_CELLS = 1.5   # 3x3 cell neighbourhood
SURFACE_MAX_RATIO = 0.4      # smaller/larger spread of a line-like neighbourhood
SURFACE_MIN_WALL_M = 1.5     # wall length that must constrain the weakest direction


def checked_pose(values):
    error = 'initial pose must contain finite x, y, yaw'
    try:
        pose = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError(error)
    return pose


def read_initial_pose(path):
    try:
        values = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(values, list) or not all(type(v) in (float, int) for v in values):
            raise ValueError('expected JSON array of numbers [x, y, yaw]')
        return checked_pose(values)
    except ValueError as exc:
        raise ValueError(f'{path}: {exc}') from exc


def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def world(xy, pose):
    return xy @ rotation(pose[2]).T + pose[:2]


def relative(a, b):
    xy = rotation(a[2]).T @ (b[:2] - a[:2])
    return np.r_[xy, np.arctan2(np.sin(b[2] - a[2]), np.cos(b[2] - a[2]))]


def predict(pose, previous, scan):
    step = np.zeros(3)
    if previous is not None and scan.odom is not None:
        step = relative(previous, scan.odom)
    if scan.gyro_delta is not None:
        step[2] = scan.gyro_delta
    return np.r_[world(step[:2], pose), pose[2] + step[2]]


def match(scan, tree, seed, radius=0.45, targets=None):
    """Trimmed point-to-point ICP. Return the prior if support is insufficient.

    ``targets`` maps matched tree indices to the points ICP aligns to; by default
    these are the tree points themselves (see ``WallSurface``).
    """
    pose = seed.copy()
    fraction = 0.0
    if len(scan) < MIN_MATCH_POINTS:
        return seed, fraction, False
    for _ in range(24):
        src = world(scan, pose)
        distances, indices = tree.query(src, distance_upper_bound=radius)
        valid = np.isfinite(distances)
        fraction = float(valid.mean())
        if fraction < 0.40 or valid.sum() < MIN_MATCH_POINTS:
            return seed, fraction, False
        keep = valid & (distances <= max(0.06, np.quantile(distances[valid], 0.85)))
        a = src[keep]
        b = tree.data[indices[keep]] if targets is None else targets(indices[keep])
        ac, bc = a.mean(axis=0), b.mean(axis=0)
        left = a - ac
        right = b - bc
        dot = (left * right).sum()
        cross = (left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]).sum()
        if np.hypot(dot, cross) < 1e-8:
            return seed, fraction, False
        angle = np.arctan2(cross, dot)
        old = pose.copy()
        pose[:2] = rotation(angle) @ (old[:2] - ac) + bc
        pose[2] += angle
        if np.linalg.norm(old[:2] - pose[:2]) < 0.0005 and abs(angle) < 0.0003:
            break
    residuals, _ = tree.query(world(scan, pose), distance_upper_bound=radius)
    # A strict post-fit check keeps an unsupported prediction out of the map.
    fraction = float((residuals < 0.16).mean())
    if fraction < 0.40 or np.linalg.norm(pose[:2] - seed[:2]) > 1.5 * radius:
        return seed, fraction, False
    return pose, fraction, True


class WallSurface:
    """Matched cell centres projected onto the local wall line, computed lazily.

    Cell centres of an occupancy grid describe a wall as a staircase that is
    offset by up to half a cell from the true surface. For a matched centre the
    occupied cells of its 3x3 neighbourhood are fitted with a line, and the
    centre moves along the line normal onto it. Along the wall the point stays
    put. Wall ends of a two-cell-thick wall, single cells and cluttered
    patches, whose neighbourhood is not line-like, keep their centre. The
    corner cell of a one-cell-thick L (three cells) still counts as line-like
    and moves diagonally by a third of a cell.

    The surface is used only for scans whose map neighbourhood constrains both
    translation directions (``constrains``). In a corridor seen at short range
    the smooth surface lets ICP slide along the walls with the odometry scale
    error, while the staircase of cell centres happens to hold it; such scans
    are matched against cell centres, exactly as without the surface.

    Only centres that ICP actually touches are processed and cached, so loading
    a large map costs the same time and memory as the plain KD-tree.
    """

    def __init__(self, tree, *, radius_cells=SURFACE_RADIUS_CELLS, max_ratio=SURFACE_MAX_RATIO,
                 min_wall_m=SURFACE_MIN_WALL_M):
        self.tree = tree
        data = tree.data
        sample = data[::max(1, len(data) // 4096)]
        spacing = np.median(tree.query(sample, k=2)[0][:, 1]) if len(data) > 1 else 0.0
        valid = np.isfinite(spacing) and spacing > 0
        self.radius = radius_cells * spacing if valid else 0.0
        self.min_cells = min_wall_m / spacing if valid else np.inf
        self.max_ratio = max_ratio
        # Processed map indices (sorted) and their rows of x, y, nx, ny.
        self.known = np.empty(0, dtype=np.intp)
        self.rows = np.empty((0, 4))

    def _patches(self, indices):
        """Rows of target x, y and unit normal (zero if not line-like) per index."""
        indices = np.asarray(indices, dtype=np.intp)
        if not self.radius:
            return np.column_stack([self.tree.data[indices], np.zeros((len(indices), 2))])
        slot = np.searchsorted(self.known, indices)
        hit = slot < len(self.known)
        hit[hit] = self.known[slot[hit]] == indices[hit]
        if not hit.all():
            data = self.tree.data
            missing = np.unique(indices[~hit])
            groups = self.tree.query_ball_point(data[missing], self.radius)
            new = np.array([wall_patch(data[i], data[ids], self.max_ratio)
                            for i, ids in zip(missing, groups)]).reshape(-1, 4)
            at = np.searchsorted(self.known, missing)
            self.known = np.insert(self.known, at, missing)
            self.rows = np.insert(self.rows, at, new, axis=0)
            slot = np.searchsorted(self.known, indices)
        return self.rows[slot]

    def __call__(self, indices):
        return self._patches(indices)[:, :2]

    def constrains(self, xy, pose, radius=0.45):
        """True if line-like map cells near the scan pin both translation directions.

        The smaller eigenvalue of the sum of n n^T over distinct matched cells
        is the number of cells, i.e. the wall length in cells, acting along the
        weakest direction. Counting distinct cells keeps a denser lidar from
        inflating it; a scan sparser than the grid counts fewer cells.
        """
        distances, indices = self.tree.query(world(xy, pose), distance_upper_bound=radius)
        normals = self._patches(np.unique(indices[np.isfinite(distances)]))[:, 2:]
        return bool(np.linalg.eigvalsh(normals.T @ normals)[0] >= self.min_cells)


def wall_patch(point, neighbours, max_ratio=SURFACE_MAX_RATIO):
    """Project ``point`` onto the line fitted to ``neighbours``; return x, y, nx, ny."""
    x, y = float(point[0]), float(point[1])
    if len(neighbours) < 3:
        return x, y, 0.0, 0.0
    mean = neighbours.mean(axis=0)
    centred = neighbours - mean
    values, axes = np.linalg.eigh(centred.T @ centred)
    if values[1] <= 0 or values[0] > max_ratio * values[1]:
        return x, y, 0.0, 0.0
    normal = axes[:, 0]
    x, y = point - ((point - mean) @ normal) * normal
    return float(x), float(y), float(normal[0]), float(normal[1])


def surface_points(points):
    """Project every map point onto its local wall line (eager form of WallSurface)."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points
    return WallSurface(cKDTree(points))(np.arange(len(points)))


@dataclass
class Estimate:
    t: float
    pose: np.ndarray
    status: str
    inliers: float
    rejection_reason: str | None = None


def track(stream, initial, *, grid=None, map_points=None, recovery=None, surface=False):
    """Build a map (``grid``) or localize against ``map_points``.

    With ``surface`` localization aligns well-constrained scans to the wall
    surface (``WallSurface``) instead of raw cell centres. It is off by default
    until it is confirmed on field data (docs/accuracy-experiment.md).
    """
    building = grid is not None
    if building == (map_points is not None):
        raise ValueError('Supply either a grid to build or an existing map')
    pose = checked_pose(initial)
    tree = None if building else cKDTree(map_points)
    targets = WallSurface(tree) if surface and tree is not None else None
    previous = None
    keyframes = deque(maxlen=18)
    keypose = None
    for scan in stream:
        predicted = predict(pose, previous, scan)
        previous = scan.odom
        pose, support, status = predicted, 0.0, 'prediction_only'
        if tree is None and len(scan.xy) >= MIN_MATCH_POINTS:
            status = 'origin'
        elif tree is not None:
            use = targets if targets is not None and targets.constrains(scan.xy, predicted) else None
            pose, support, accepted = match(scan.xy, tree, predicted, targets=use)
            if accepted:
                status = 'matched'
        if recovery is not None:
            recovered = recovery.update(scan, status == 'matched')
            if recovered is not None:
                # Score this accepted relocalization against exactly the exported map.
                pose = recovered
                distances = tree.query(world(scan.xy, pose))[0]
                support = float(np.mean(distances < .16))
                status = 'matched'
        if building and status != 'prediction_only':
            points = world(scan.xy, pose)
            grid.observe(world(scan.origin, pose), points)
            if (keypose is None or np.linalg.norm(pose[:2] - keypose[:2]) >= 0.15
                    or abs(pose[2] - keypose[2]) >= 0.12):
                keyframes.append(points)
                joined = np.vstack(keyframes)
                _, ids = np.unique(np.floor(joined / 0.06), axis=0, return_index=True)
                tree = cKDTree(joined[ids])
                keypose = pose.copy()
        reason = None
        if status == 'prediction_only':
            if not len(scan.xy):
                reason = getattr(scan, 'empty_reason', None) or 'no_usable_points'
            elif len(scan.xy) < MIN_MATCH_POINTS:
                reason = 'too_few_points'
            else:
                reason = 'icp_rejected'
        yield Estimate(scan.t, pose.copy(), status, support, reason)
