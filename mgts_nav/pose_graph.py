"""Offline endpoint loop constraint on a planar keyframe pose graph."""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .slam import relative, world


def adjust(trajectory, endpoint_measurement):
    """Anchor node 0, preserve local increments, distribute a measured closure."""
    poses = np.asarray(trajectory, dtype=float)
    measured = np.asarray(endpoint_measurement, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 3 or len(poses) < 3 or measured.shape != (3,):
        raise ValueError('Expected at least three poses and a relative endpoint constraint')
    if not np.isfinite(poses).all() or not np.isfinite(measured).all():
        raise ValueError('Nonfinite pose graph')
    # Space keyframes by accumulated motion so a pause receives no extra weight.
    turns = np.diff(np.unwrap(poses[:, 2]))
    distances = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    progress = np.r_[0, np.cumsum(distances + 0.4 * abs(turns))]
    if progress[-1] < 0.1:
        raise ValueError('No observable motion in loop')
    selected = np.unique(np.r_[0, np.searchsorted(progress, np.linspace(0, progress[-1], 48)), len(poses)-1])
    # A stationary tail is represented by its last scan.
    selected = selected[np.r_[np.diff(progress[selected]) > 1e-8, True]]
    selected[0] = 0
    base = poses[selected].copy()
    base[:, 2] = np.unwrap(base[:, 2])
    increments = np.array([relative(a, b) for a, b in zip(base[:-1], base[1:])])

    def residual(flat):
        candidate = np.vstack([base[0], flat.reshape(-1, 3)])
        d = np.diff(candidate, axis=0)
        theta = candidate[:-1, 2]
        c, s = np.cos(theta), np.sin(theta)
        local = np.column_stack([c*d[:, 0] + s*d[:, 1], -s*d[:, 0] + c*d[:, 1], d[:, 2]])
        err = np.vstack([local - increments, relative(candidate[0], candidate[-1]) - measured])
        err[:, 2] = np.arctan2(np.sin(err[:, 2]), np.cos(err[:, 2]))
        return (err / [0.06, 0.06, 0.04]).ravel()

    edges = len(base) - 1
    pattern = lil_matrix((3*(edges+1), 3*edges), dtype=int)
    for i in range(edges):
        pattern[3*i:3*i+3, 3*max(0, i-1):3*(i+1)] = 1
    pattern[-3:, -3:] = 1
    solution = least_squares(residual, base[1:].ravel(), jac_sparsity=pattern.tocsr(),
                             max_nfev=60)
    if not solution.success:
        raise ValueError(f'Pose graph optimization failed: {solution.message}')
    fitted = np.vstack([base[0], solution.x.reshape(-1, 3)])
    # Interpolate map corrections in each keyframe's local frame.
    corrections = np.array([relative(a, b) for a, b in zip(base, fitted)])
    corrections[:, 2] = np.unwrap(corrections[:, 2])
    interp = np.column_stack([np.interp(progress, progress[selected], col) for col in corrections.T])
    adjusted = poses.copy()
    for i, delta in enumerate(interp):
        adjusted[i, :2] = world(delta[:2], poses[i])
    adjusted[:, 2] += interp[:, 2]
    return adjusted
