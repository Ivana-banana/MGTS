"""Point-to-plane ICP against a 3D point map with robust weights and degeneracy handling."""
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from . import se3


def estimate_normals(points, tree, k=10, max_radius=0.5):
    """Normals and planarity (0..1) from the k nearest neighbours of every point.

    Points whose neighbourhood is too small or too spread get planarity 0 and are
    not used as plane targets.
    """
    n = len(points)
    normals = np.zeros((n, 3))
    planarity = np.zeros(n)
    k = min(k, n)
    if k < 4:
        return normals, planarity
    chunk = 50000
    for s in range(0, n, chunk):
        q = points[s:s + chunk]
        d, idx = tree.query(q, k=k)
        nb = points[idx]
        mu = nb.mean(axis=1, keepdims=True)
        c = nb - mu
        cov = np.einsum('nki,nkj->nij', c, c) / k
        w, v = np.linalg.eigh(cov)
        normals[s:s + chunk] = v[:, :, 0]
        w = np.maximum(w, 0)
        # planarity: (l2 - l1) / l3 with l1 <= l2 <= l3
        pl = (w[:, 1] - w[:, 0]) / np.maximum(w[:, 2], 1e-12)
        pl[d[:, -1] > max_radius] = 0.0
        planarity[s:s + chunk] = pl
    return normals, planarity


class PointMap:
    """Static registration target: points, KD-tree and plane normals."""

    def __init__(self, points, k_normals=10, min_planarity=0.1):
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        self.tree = cKDTree(self.points)
        self.normals, self.planarity = estimate_normals(self.points, self.tree, k=k_normals)
        self.usable = self.planarity >= min_planarity

    def __len__(self):
        return len(self.points)


@dataclass
class IcpResult:
    transform: np.ndarray
    iterations: int
    converged: bool
    n_points: int
    n_matched: int
    inlier_ratio: float
    rmse: float
    eigenvalues: np.ndarray
    degenerate_dims: int
    information: np.ndarray = field(repr=False)


def prior_information(sigma_rot, sigma_trans, sigma_point=0.05):
    """Diagonal prior information in the units of the ICP normal equations.

    Point residuals enter the normal equations unscaled, i.e. as if their sigma
    were ``sigma_point``; the prior is scaled to the same units. ``None`` or a
    non-positive sigma disables the prior on that block.
    """
    lam = np.zeros(6)
    if sigma_rot and sigma_rot > 0:
        lam[:3] = (sigma_point / sigma_rot) ** 2
    if sigma_trans and sigma_trans > 0:
        lam[3:] = (sigma_point / sigma_trans) ** 2
    return lam


def _body_error(t_ref, t):
    """Right-perturbation error e with t = t_ref * Exp(e), e = (rot, trans) in the body frame."""
    rr = t_ref[:3, :3].T @ t[:3, :3]
    return np.concatenate([se3.so3_log(rr), t_ref[:3, :3].T @ (t[:3, 3] - t_ref[:3, 3])])


def register(src, target, init, max_iterations=20, corr_schedule=(0.8, 0.5, 0.3), robust_scale=0.08,
             inlier_threshold=0.1, degeneracy_ratio=1e-3, min_matches=30, eps=1e-4, prior=None):
    """Align source points (sensor/base frame) to ``target`` starting from ``init``.

    Returns IcpResult; ``transform`` maps source into the target frame. The
    update is a right (body-frame) perturbation, so rotation and translation
    are expressed about the robot, not about the map origin.

    ``prior`` (6 values, rotation then translation, see prior_information)
    pulls the estimate towards ``init``: this is the odometry prediction, and
    the result is the maximum a posteriori pose given the prediction and the
    scan. Without a prior, directions whose Hessian eigenvalue is below
    ``degeneracy_ratio`` * largest eigenvalue are frozen at ``init``, so a
    featureless corridor cannot pull the pose along its axis.
    """
    t = init.copy()
    lam = np.zeros(6) if prior is None else np.asarray(prior, dtype=float)
    n_src = len(src)
    eigvals = np.zeros(6)
    info = np.zeros((6, 6))
    degenerate = 0
    converged = False
    it = 0
    for it in range(1, max_iterations + 1):
        stage = min(len(corr_schedule) - 1, (it - 1) * len(corr_schedule) // max_iterations)
        max_corr = corr_schedule[stage]
        rot = t[:3, :3]
        q = src @ rot.T + t[:3, 3]
        d, idx = target.tree.query(q, distance_upper_bound=max_corr)
        ok = np.isfinite(d)
        idx_ok = idx[ok]
        ok_plane = target.usable[idx_ok]
        sel = np.flatnonzero(ok)[ok_plane]
        if len(sel) < min_matches:
            break
        nidx = idx[sel]
        nrm = target.normals[nidx]
        r = np.einsum('ij,ij->i', nrm, q[sel] - target.points[nidx])
        w = 1.0 / (1.0 + (r / robust_scale) ** 2)  # Cauchy kernel
        nb = nrm @ rot                              # normals in the body frame
        jac = np.hstack([np.cross(src[sel], nb), nb])
        h = jac.T @ (jac * w[:, None])
        g = jac.T @ (w * r)
        eigvals = np.linalg.eigvalsh(h)
        info = h
        if lam.any():
            e = _body_error(init, t)
            delta = -np.linalg.solve(h + np.diag(lam), g + lam * e)
            degenerate = int((eigvals < degeneracy_ratio * max(eigvals[-1], 1e-12)).sum())
        else:
            ev, vecs = np.linalg.eigh(h)
            good = ev > degeneracy_ratio * max(ev[-1], 1e-12)
            degenerate = int(6 - good.sum())
            coef = np.zeros(6)
            coef[good] = -(vecs[:, good].T @ g) / ev[good]
            delta = vecs @ coef
        t = t @ se3.make(se3.so3_exp(delta[:3]), delta[3:])
        if np.linalg.norm(delta[:3]) < eps and np.linalg.norm(delta[3:]) < eps and stage == len(corr_schedule) - 1:
            converged = True
            break
    # final statistics at the tightest gate
    q = se3.apply(t, src)
    d, idx = target.tree.query(q, distance_upper_bound=corr_schedule[-1])
    ok = np.isfinite(d)
    n_matched = int(ok.sum())
    if n_matched:
        nidx = idx[ok]
        plane_ok = target.usable[nidx]
        r = np.abs(np.einsum('ij,ij->i', target.normals[nidx], q[ok] - target.points[nidx]))
        # point-to-point distance for non-planar targets
        r = np.where(plane_ok, r, d[ok])
        inl = r < inlier_threshold
        inlier_ratio = float(inl.sum()) / max(n_src, 1)
        rmse = float(np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else float('nan')
    else:
        inlier_ratio, rmse = 0.0, float('nan')
    return IcpResult(t, it, converged, n_src, n_matched, inlier_ratio, rmse, eigvals, degenerate, info)


def residuals_to_map(points_in_map, target, gate=0.3):
    """Distance of each point to the map (plane distance where available); inf beyond gate."""
    d, idx = target.tree.query(points_in_map, distance_upper_bound=gate)
    out = np.full(len(points_in_map), np.inf)
    ok = np.isfinite(d)
    if ok.any():
        nidx = idx[ok]
        plane = np.abs(np.einsum('ij,ij->i', target.normals[nidx], points_in_map[ok] - target.points[nidx]))
        out[ok] = np.where(target.usable[nidx], np.minimum(plane, d[ok]), d[ok])
    return out
