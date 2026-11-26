"""Planar scan matching in map coordinates, with body-frame motion increments."""
from collections import deque
from dataclasses import dataclass
import json

import numpy as np
from scipy.spatial import cKDTree


MIN_MATCH_POINTS = 12
MATCH_RADIUS_M = .45


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


def match(scan, tree, seed, radius=MATCH_RADIUS_M):
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


def point_to_line_rows(offset, translation, centers, normals, lever):
    residual = np.sum(normals * (offset + translation - centers), axis=1)
    yaw = (-normals[:, 0] * offset[:, 1] + normals[:, 1] * offset[:, 0]) / lever
    return residual, np.column_stack((normals, yaw))


class PointToLine:
    """Fit a fixed map while using the predicted pose as a prior."""

    def __init__(self, tree):
        self.tree = tree
        _, neighbors = tree.query(tree.data, k=8)
        patches = tree.data[neighbors]
        self.centers = patches.mean(axis=1)
        offsets = patches - self.centers[:, None]
        values, axes = np.linalg.eigh(offsets.transpose(0, 2, 1) @ offsets)
        self.normals = axes[:, :, 0]
        self.linear = (values[:, 0] <= .12 * values[:, 1]) & (values[:, 1] > 0)

    def refine(self, scan, seed, predicted, *, odom_translation=False):
        # Express translation and rotation in metres using the scan RMS range.
        lever = np.sqrt(np.mean(np.sum(scan ** 2, axis=1)))
        scale = np.array([1., 1., lever])
        prior = np.diag([.10, .10, .50])
        pose = seed.copy()
        for _ in range(8):
            offset = scan @ rotation(pose[2]).T
            distances, indices = self.tree.query(offset + pose[:2])
            keep = (distances < .16) & self.linear[indices]
            if np.count_nonzero(keep) < 12:
                return None
            ids = indices[keep]
            residual, jacobian = point_to_line_rows(
                offset[keep], pose[:2], self.centers[ids], self.normals[ids], lever)
            # Huber weights with a 4 cm threshold.
            weights = .04 / np.maximum(.04, np.abs(residual))
            weighted_transpose = jacobian.T * weights / len(weights)
            delta = (pose - seed) * scale
            if odom_translation:
                # Cap the radial influence of translation from odometry at 1 cm.
                shift = np.linalg.norm(delta[:2])
                prior[0, 0] = prior[1, 1] = .10 if shift <= .01 else .10 * .01 / shift
            gradient = weighted_transpose @ residual + prior @ delta
            step = np.linalg.solve(weighted_transpose @ jacobian + prior, -gradient) / scale
            pose += step
            if np.linalg.norm(step[:2]) < .0001 and abs(step[2]) < .0001:
                break
        support = float(np.mean(self.tree.query(world(scan, pose))[0] < .16))
        if support >= .40 and np.linalg.norm(pose[:2] - predicted[:2]) <= 1.5 * MATCH_RADIUS_M:
            return pose, support
        return None


@dataclass
class Estimate:
    t: float
    pose: np.ndarray
    status: str
    inliers: float
    rejection_reason: str | None = None


def track(stream, initial, *, grid=None, map_points=None, recovery=None, velocity_prior=None, drift_recovery=None):
    building = grid is not None
    if building == (map_points is not None):
        raise ValueError('Supply either a grid to build or an existing map')
    if velocity_prior is not None and not building:
        raise ValueError("Velocity prior requires a mapping grid")
    predict_step = predict if velocity_prior is None else velocity_prior.predict
    pose = checked_pose(initial)
    tree = None if building else cKDTree(map_points)
    refinement = PointToLine(tree) if tree is not None and tree.n >= 8 else None
    previous = None
    previous_t = None
    previous_fit = False
    velocity_bias = np.zeros(2)
    keyframes = deque(maxlen=18)
    keypose = None
    for scan in stream:
        raw_prediction = predict_step(pose, previous, scan)
        predicted = raw_prediction
        odom_step = previous is not None and scan.odom is not None
        has_translation = scan.odom is not None and (odom_step or previous_t is None)
        dt = 0. if previous_t is None else scan.t - previous_t
        if dt <= 0.:
            velocity_bias.fill(0.)
        previous = scan.odom
        # Bias is only for coasting without lidar. Feeding it into ICP seeds made
        # Go2 diverge from good odometry (1250 → 862 matches, multi-metre end error).
        usable_lidar = len(scan.xy) >= MIN_MATCH_POINTS
        if not usable_lidar and dt > 0. and not has_translation:
            speed = np.linalg.norm(velocity_bias)
            if speed > .06:
                predicted = raw_prediction.copy()
                predicted[:2] += dt * (1. - .06 / speed) * velocity_bias
        pose, support, status = predicted, 0.0, 'prediction_only'
        refined = None
        if tree is None and usable_lidar:
            status = 'origin'
        elif tree is not None:
            pose, support, accepted = match(scan.xy, tree, predicted)
            if accepted:
                # Point-to-line with odometry poisoned Go2 corridors (1250→862 matched).
                # Keep it for lidar-only / gyro translation; with measured odom stay on ICP.
                if not has_translation:
                    seed = pose
                    if scan.gyro_delta is not None:
                        seed = np.r_[pose[:2], predicted[2]]
                    candidate = (refinement.refine(scan.xy, seed, predicted, odom_translation=False)
                                 if refinement is not None else None)
                    if candidate is not None and candidate[1] + 1e-9 >= support:
                        pose, support = candidate
                        refined = candidate
                        if previous_fit and dt > 0.:
                            # Lidar vs raw prediction bias, used only while coasting.
                            measured = (pose[:2] - raw_prediction[:2]) / dt
                            velocity_bias += -np.expm1(-dt / .4) * (measured - velocity_bias)
                status = 'matched'
        if recovery is not None:
            recovered = recovery.update(scan, status == 'matched')
            if recovered is not None:
                # Score this accepted relocalization against exactly the exported map.
                pose = recovered
                distances = tree.query(world(scan.xy, pose))[0]
                support = float(np.mean(distances < .16))
                status = 'matched'
                velocity_bias.fill(0.)
                refined = None
        if drift_recovery is not None:
            confirmed = drift_recovery.update(scan, pose, support, status == 'matched', tree)
            if confirmed is not None:
                pose = confirmed
                support = float(np.mean(tree.query(world(scan.xy, pose))[0] < .16))
                status = 'matched'
                velocity_bias.fill(0.)
                refined = None
            elif drift_recovery.uncertain:
                status = 'uncertain'
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
        previous_t = scan.t
        previous_fit = refined is not None
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
