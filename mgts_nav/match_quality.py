"""Grade a 3D scan-to-map fit: confident, weak or prediction_only (#214).

The criteria were fixed in the issue before they were checked against revisits:

* overlap   - share of held-out points within INLIER_M of the map at the fit, >= MIN_INLIERS;
* improves  - median held-out residual at the fit <= the one at the prior + IMPROVE_SLACK_M;
* observable - smallest / largest eigenvalue of the weighted point-to-plane system at the
  fit >= MIN_EIGEN_RATIO, the same scaling and threshold as SurfaceMap._align;
* small     - correction against the prior <= MAX_SHIFT_M and <= MAX_TURN_RAD.

An accepted fit meeting all four is ``confident``; otherwise ``weak`` with the failed
criteria listed.  A rejected fit is ``prediction_only``.
"""
from dataclasses import dataclass, field

import numpy as np

from .localize3d import MAX_MATCH_M

INLIER_M = 0.10
MIN_INLIERS = 0.60
IMPROVE_SLACK_M = 0.005
MIN_EIGEN_RATIO = 0.008
MAX_SHIFT_M = 0.30
MAX_TURN_RAD = 0.10


@dataclass
class Quality:
    level: str
    reasons: list = field(default_factory=list)
    inliers: float = float('nan')
    residual_prior_m: float = float('nan')
    residual_fit_m: float = float('nan')
    eigen_ratio: float = float('nan')
    weakest_axis: str = ''
    shift_m: float = float('nan')
    turn_rad: float = float('nan')


AXES = ('x', 'y', 'z', 'roll', 'pitch', 'yaw')


def eigen_ratio(surface, points, pose):
    """Conditioning of the point-to-plane system at `pose` and its weakest direction."""
    rotated = pose.rotation.apply(points)
    world = rotated + pose.xyz
    distances, indices = surface.tree.query(world)
    mask = (distances < MAX_MATCH_M) & surface.planar[indices]
    if np.count_nonzero(mask) < 6:
        return 0.0, ''
    normal = surface.normals[indices[mask]]
    error = np.einsum('ij,ij->i', world[mask] - surface.tree.data[indices[mask]], normal)
    scale = max(1., np.median(np.linalg.norm(rotated[mask], axis=1)))
    jacobian = np.column_stack((normal, np.cross(rotated[mask], normal) / scale))
    weight = 1 / np.sqrt(1 + (error / 0.08) ** 2)
    system = jacobian * weight[:, None]
    values, vectors = np.linalg.eigh(system.T @ system)
    if values[-1] <= 0:
        return 0.0, ''
    return float(values[0] / values[-1]), AXES[int(np.argmax(np.abs(vectors[:, 0])))]


def assess(surface, fit_points, held_out, prior, fit, accepted):
    if not accepted:
        return Quality('prediction_only', ['icp_rejected'])
    d_fit = surface.tree.query(fit.apply(held_out))[0]
    d_prior = surface.tree.query(prior.apply(held_out))[0]
    ratio, axis = eigen_ratio(surface, fit_points, fit)
    q = Quality('confident', [], float(np.mean(d_fit < INLIER_M)), float(np.median(d_prior)),
                float(np.median(d_fit)), ratio, axis, float(np.linalg.norm(fit.xyz - prior.xyz)),
                float((fit.rotation * prior.rotation.inv()).magnitude()))
    if q.inliers < MIN_INLIERS:
        q.reasons.append('overlap')
    if q.residual_fit_m > q.residual_prior_m + IMPROVE_SLACK_M:
        q.reasons.append('no_improvement')
    if q.eigen_ratio < MIN_EIGEN_RATIO:
        q.reasons.append('degenerate')
    if q.shift_m > MAX_SHIFT_M or q.turn_rad > MAX_TURN_RAD:
        q.reasons.append('large_correction')
    if q.reasons:
        q.level = 'weak'
    return q
