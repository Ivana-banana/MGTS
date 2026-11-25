"""Planar scan matching in map coordinates, with body-frame motion increments."""
from collections import deque
from dataclasses import dataclass
import json

import numpy as np
from scipy.spatial import cKDTree


MIN_MATCH_POINTS = 12


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


class VelocityPrior:
    """Bounded body-frame translation from consecutive ICP matches during mapping."""

    def __init__(self):
        self.history = deque(maxlen=2)
        self.applied = 0

    def predict(self, pose, previous, scan):
        result = predict(pose, previous, scan)
        if scan.odom is not None or (self.history and not 0 < scan.t - self.history[-1].t <= .25):
            self.history.clear()
        if len(self.history) == 2:
            a, b = self.history
            dt = scan.t - b.t
            step = relative(a.pose, b.pose)[:2] * (dt / (b.t - a.t))
            length = np.linalg.norm(step)
            limit = min(1.5 * dt, .20)
            if length > limit:
                step *= limit / length
            result[:2] = world(step, b.pose)
            self.applied += 1
        return result

    def observe(self, row):
        if row.status == "matched":
            self.history.append(row)
        else:
            self.history.clear()


def match(scan, tree, seed, radius=0.45):
    """Trimmed point-to-point ICP. Return the prior if support is insufficient."""
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
        a, b = src[keep], tree.data[indices[keep]]
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


@dataclass
class Estimate:
    t: float
    pose: np.ndarray
    status: str
    inliers: float
    rejection_reason: str | None = None


def track(stream, initial, *, grid=None, map_points=None, recovery=None, velocity_prior=None):
    building = grid is not None
    if building == (map_points is not None):
        raise ValueError('Supply either a grid to build or an existing map')
    if velocity_prior is not None and not building:
        raise ValueError("Velocity prior requires a mapping grid")
    predict_step = predict if velocity_prior is None else velocity_prior.predict
    pose = checked_pose(initial)
    tree = None if building else cKDTree(map_points)
    previous = None
    keyframes = deque(maxlen=18)
    keypose = None
    for scan in stream:
        predicted = predict_step(pose, previous, scan)
        previous = scan.odom
        pose, support, status = predicted, 0.0, 'prediction_only'
        if tree is None and len(scan.xy) >= MIN_MATCH_POINTS:
            status = 'origin'
        elif tree is not None:
            pose, support, accepted = match(scan.xy, tree, predicted)
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
        estimate = Estimate(scan.t, pose.copy(), status, support, reason)
        if velocity_prior is not None:
            velocity_prior.observe(estimate)
        yield estimate
