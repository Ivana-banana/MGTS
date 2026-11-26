"""Offline 3D map building from the mapping bag.

Pipeline:
1. deskew every sweep with odometry (point time = stamp + ``time`` field);
2. pass 1: sequential scan-to-map ICP with the odometry increment as prediction.
   The scan is registered to everything mapped so far, so a revisited place is
   aligned to its first observation (implicit loop closure);
3. pass 2: every scan is re-registered to the map built from pass-1 poses;
4. dynamic-object removal: voxels hit in too few scans or crossed by many
   lidar rays (free-space carving) are removed;
5. the result is voxelized (centroids) and saved as PCD plus a 2D slice.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from . import cloud, se3
from .deskew import deskew
from .icp import PointMap, prior_information, register


@dataclass
class MapParams:
    map_voxel: float = 0.05          # output map resolution, m
    reg_voxel: float = 0.10          # scan downsampling before ICP, m
    min_range: float = 0.3
    max_range: float = 12.0
    rebuild_every: int = 10          # pass 1: rebuild KD-tree every N accepted scans
    min_inlier_ratio: float = 0.35   # accept a registration only above this
    max_correction_m: float = 0.5    # ... and when it moves the prediction less than this
    max_correction_rad: float = 0.35
    passes: int = 2
    dyn_voxel: float = 0.10          # voxel for hit/free statistics
    dyn_min_scans: int = 3           # a voxel must be hit in at least this many sweeps
    dyn_free_ratio: float = 0.75     # remove if free / (free + hits) above this ...
    dyn_min_free: int = 6            # ... and crossed by at least this many rays
    carve_margin: float = 0.25       # stop carving this far before each return, m
    carve_rays: int = 256            # rays per sweep used for carving
    sensor_origin: tuple = (0.2822, 0.0, -0.0247)  # lidar origin in base_link (estimated from the bag)
    prior_sigma_rot: float = 0.01    # odometry prediction uncertainty per sweep, rad (<=0 disables)
    prior_sigma_trans: float = 0.02  # ... and m
    planar: bool = True              # correct only x, y, yaw of map->odom (one flat floor;
                                     # roll/pitch come from gravity, z from leg kinematics)


@dataclass
class MapResult:
    points: np.ndarray
    intensity: np.ndarray
    poses: list                      # T_map_base per scan (refined)
    odom_poses: list                 # T_odom_base per scan
    stamps: np.ndarray
    stats: dict = field(default_factory=dict)


def prepare_scans(scans, odom, params):
    """Deskew and range-filter scans; returns list of (stamp, pts_base, intensity, T_odom_base)."""
    out = []
    skipped = 0
    for s in scans:
        m = cloud.valid_mask(s.points, params.min_range, params.max_range)
        pts, tms, inten = s.points[m], s.times[m], s.intensity[m]
        try:
            pts = deskew(pts, tms, s.stamp, odom)
            t_ob = odom.at(s.stamp)
        except LookupError:
            skipped += 1
            continue
        if t_ob is None or len(pts) < 50:
            skipped += 1
            continue
        out.append((s.stamp, pts, inten, t_ob))
    return out, skipped


class _VoxelAccumulator:
    def __init__(self, size):
        self.size = size
        self.sums = {}

    def add(self, pts):
        keys = cloud.pack_keys(cloud.voxel_keys(pts, self.size))
        uk, inv = np.unique(keys, return_inverse=True)
        inv = inv.reshape(-1)
        s = np.zeros((len(uk), 3))
        np.add.at(s, inv, pts)
        c = np.bincount(inv, minlength=len(uk))
        for k, ss, cc in zip(uk.tolist(), s, c):
            prev = self.sums.get(k)
            if prev is None:
                self.sums[k] = [ss, cc]
            else:
                prev[0] = prev[0] + ss
                prev[1] += cc

    def points(self):
        if not self.sums:
            return np.zeros((0, 3))
        return np.array([v[0] / v[1] for v in self.sums.values()])


def _prior(params):
    lam = prior_information(params.prior_sigma_rot, params.prior_sigma_trans)
    if params.planar:
        lam = np.where(se3.planar_prior_mask(), 1e8, lam)
    return lam


def _constrain(pose, t_ob, params):
    """Project the map->odom correction implied by ``pose`` onto x, y, yaw when planar."""
    if not params.planar:
        return pose
    return se3.planar(pose @ se3.inv(t_ob)) @ t_ob


def _accept(res, pred, params):
    if res.inlier_ratio < params.min_inlier_ratio:
        return False
    dt, dr = se3.delta_norm(se3.inv(pred) @ res.transform)
    return dt <= params.max_correction_m and dr <= params.max_correction_rad


def sequential_pass(prepared, params, log=print):
    acc = _VoxelAccumulator(params.map_voxel)
    poses = []
    target = None
    since_rebuild = 0
    accepted = 0
    prev_corr, prev_odom = None, None
    for i, (stamp, pts, _, t_ob) in enumerate(prepared):
        if prev_corr is None:
            pose = t_ob.copy()  # map frame := odom frame of the mapping bag
        else:
            pred = _constrain(prev_corr @ se3.inv(prev_odom) @ t_ob, t_ob, params)
            pose = pred
            if target is not None:
                src = cloud.voxel_downsample(pts, params.reg_voxel)
                res = register(src, target, pred, prior=_prior(params))
                if _accept(res, pred, params):
                    pose = _constrain(res.transform, t_ob, params)
                    accepted += 1
        acc.add(se3.apply(pose, pts))
        since_rebuild += 1
        if target is None or since_rebuild >= params.rebuild_every:
            target = PointMap(acc.points())
            since_rebuild = 0
        poses.append(pose)
        prev_corr, prev_odom = pose, t_ob
        if log and i % 200 == 0:
            log('  pass1 scan %d/%d map %d voxels accepted %d' % (i, len(prepared), len(acc.sums), accepted))
    return poses, accepted


def refine_pass(prepared, poses, params):
    acc = _VoxelAccumulator(params.map_voxel)
    for (stamp, pts, _, _), pose in zip(prepared, poses):
        acc.add(se3.apply(pose, pts))
    target = PointMap(acc.points())
    out = []
    accepted = 0
    for (stamp, pts, _, t_ob), pose in zip(prepared, poses):
        src = cloud.voxel_downsample(pts, params.reg_voxel)
        res = register(src, target, pose, prior=_prior(params))
        if _accept(res, pose, params):
            out.append(_constrain(res.transform, t_ob, params))
            accepted += 1
        else:
            out.append(pose)
    return out, accepted


def dynamic_filter(prepared, poses, params):
    """Return (packed voxel keys to keep, hit counts, free counts) at ``dyn_voxel``."""
    size = params.dyn_voxel
    hit_keys = []
    free_keys = []
    origin_b = np.array(params.sensor_origin)
    rng = np.random.default_rng(0)
    for (stamp, pts, _, _), pose in zip(prepared, poses):
        pw = se3.apply(pose, pts)
        hk = np.unique(cloud.pack_keys(cloud.voxel_keys(pw, size)))
        hit_keys.append(hk)
        o = se3.apply(pose, origin_b[None])[0]
        sel = pw if len(pw) <= params.carve_rays else pw[rng.choice(len(pw), params.carve_rays, replace=False)]
        vec = sel - o
        dist = np.linalg.norm(vec, axis=1)
        reach = dist - params.carve_margin
        good = reach > size
        if not good.any():
            continue
        vec, dist, reach = vec[good], dist[good], reach[good]
        dirs = vec / dist[:, None]
        step = size * 0.5
        nsteps = int(np.ceil(reach.max() / step))
        ts = np.arange(1, nsteps + 1) * step
        samples = o + dirs[:, None, :] * ts[None, :, None]
        mask = ts[None, :] < reach[:, None]
        fk = np.unique(cloud.pack_keys(cloud.voxel_keys(samples[mask], size)))
        free_keys.append(fk)
    hk_all, hit_counts = np.unique(np.concatenate(hit_keys), return_counts=True)
    if free_keys:
        fk_all, free_counts_u = np.unique(np.concatenate(free_keys), return_counts=True)
        pos = np.searchsorted(fk_all, hk_all)
        pos = np.clip(pos, 0, len(fk_all) - 1)
        free_counts = np.where(fk_all[pos] == hk_all, free_counts_u[pos], 0)
    else:
        free_counts = np.zeros_like(hit_counts)
    ratio = free_counts / np.maximum(free_counts + hit_counts, 1)
    dynamic = (ratio > params.dyn_free_ratio) & (free_counts >= params.dyn_min_free)
    keep = (hit_counts >= params.dyn_min_scans) & ~dynamic
    return hk_all[keep], {'voxels_total': int(len(hk_all)),
                          'removed_rare': int((hit_counts < params.dyn_min_scans).sum()),
                          'removed_free_space': int((dynamic & (hit_counts >= params.dyn_min_scans)).sum())}


def assemble(prepared, poses, params, keep_keys=None):
    allp, alli = [], []
    for (stamp, pts, inten, _), pose in zip(prepared, poses):
        allp.append(se3.apply(pose, pts))
        alli.append(inten)
    p = np.vstack(allp)
    it = np.concatenate(alli)
    if keep_keys is not None:
        k = cloud.pack_keys(cloud.voxel_keys(p, params.dyn_voxel))
        idx = np.searchsorted(keep_keys, k)
        idx = np.clip(idx, 0, max(len(keep_keys) - 1, 0))
        m = keep_keys[idx] == k if len(keep_keys) else np.zeros(len(k), bool)
        p, it = p[m], it[m]
    p, it = cloud.voxel_downsample(p, params.map_voxel, it[:, None])
    return p, it[:, 0]


def build(scans, odom, params=None, log=print):
    params = params or MapParams()
    t0 = time.perf_counter()
    prepared, skipped = prepare_scans(scans, odom, params)
    if not prepared:
        raise RuntimeError('no usable scans (skipped %d)' % skipped)
    stats = {'scans_in': len(scans), 'scans_used': len(prepared), 'scans_skipped': skipped}
    odom_poses = [p[3] for p in prepared]
    if params.passes <= 0:  # odometry-only map, kept for comparison
        poses = list(odom_poses)
    else:
        poses, acc1 = sequential_pass(prepared, params, log)
        stats['pass1_accepted'] = acc1
    for k in range(1, params.passes):
        poses, acc = refine_pass(prepared, poses, params)
        stats['pass%d_accepted' % (k + 1)] = acc
        if log:
            log('  refine pass %d accepted %d/%d' % (k + 1, acc, len(prepared)))
    keep, dyn = dynamic_filter(prepared, poses, params)
    stats.update(dyn)
    pts, inten = assemble(prepared, poses, params, keep)
    stats['map_points'] = int(len(pts))
    stats['seconds'] = round(time.perf_counter() - t0, 2)
    stamps = np.array([p[0] for p in prepared])
    return MapResult(pts, inten, poses, odom_poses, stamps, stats), prepared


def consistency(prepared, poses, params, sample_every=5, gate=0.3):
    """Leave-out check: residuals of scans to a map built from the other half of the scans.

    Scans with even index (in blocks of ``sample_every``) build the map, the
    others are measured against it, so a scan never meets its own points.
    """
    from .icp import residuals_to_map
    build_idx = [i for i in range(len(prepared)) if (i // sample_every) % 2 == 0]
    test_idx = [i for i in range(len(prepared)) if (i // sample_every) % 2 == 1]
    pts = np.vstack([se3.apply(poses[i], prepared[i][1]) for i in build_idx])
    target = PointMap(cloud.voxel_downsample(pts, params.map_voxel))
    res = np.concatenate([residuals_to_map(se3.apply(poses[i], prepared[i][1]), target, gate) for i in test_idx])
    fin = np.isfinite(res)
    return {'median_m': float(np.median(res[fin])), 'p90_m': float(np.percentile(res[fin], 90)),
            'within_5cm': float((res < 0.05).mean()), 'within_10cm': float((res < 0.10).mean()),
            'unmatched_beyond_gate': float((~fin).mean())}
