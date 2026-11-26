"""Search map-frame XY and yaw for the first Go2 base pose with a fixed gravity and height prior."""
import argparse
from itertools import islice
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from mgts_nav.localize3d import scans, voxel_centroids


def rotate(points, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return points @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def separated(a, b):
    angle = abs(np.arctan2(np.sin(a[2] - b[2]), np.cos(a[2] - b[2])))
    return np.linalg.norm(a[:2] - b[:2]) > 0.8 or angle > np.deg2rad(25)


def endpoints(rows, origin):
    points = voxel_centroids(np.concatenate([pose.apply(cloud) for _, cloud, pose, _ in rows]), .14)
    # Horizontal floor alone carries little information about XY or yaw.
    return points[points[:, 2] > origin[2] - .15] - [origin[0], origin[1], 0]


def placed(points, candidate):
    return rotate(points, candidate[2]) + [candidate[0], candidate[1], 0]


def search(fixed, points, odom_seed):
    if len(points) < 30:
        raise ValueError('Insufficient elevated endpoints for a start')
    res = .15
    lower = fixed.min(axis=0) - 1
    indices = np.rint((fixed - lower) / res).astype(int)
    empty = np.ones(indices.max(axis=0) + 8, dtype=bool)
    empty[tuple(indices.T)] = False
    field = np.minimum(distance_transform_edt(empty) * res, .5)
    x, y = [np.arange(np.floor(fixed[:, i].min()), np.ceil(fixed[:, i].max()) + .01, .5) for i in (0, 1)]
    centers = np.array(np.meshgrid(x, y)).reshape(2, -1).T
    centers3 = np.column_stack((centers, np.zeros(len(centers))))
    rng = np.random.default_rng(188)
    sample = points[rng.choice(len(points), min(256, len(points)), replace=False)]
    candidates = []
    for deg in range(-180, 180, 10):
        yaw = np.deg2rad(deg)
        rotated = rotate(sample, yaw)
        for group in np.array_split(centers3, 10):
            ijk = np.rint((group[:, None, :] + rotated - lower) / res).astype(int)
            valid = np.all((ijk >= 0) & (ijk < field.shape), axis=2)
            safe = np.clip(ijk, 0, np.array(field.shape) - 1)
            scores = np.where(valid, field[tuple(np.moveaxis(safe, -1, 0))], .5).mean(axis=1)
            candidates.extend((v, np.array([a, b, yaw])) for v, (a, b, _) in zip(scores, group))
    seeds = [odom_seed]
    for _, candidate in sorted(candidates, key=lambda row: row[0]):
        if all(separated(candidate, seed) for seed in seeds):
            seeds.append(candidate)
            if len(seeds) == 24:
                break
    tree = cKDTree(fixed)

    def loss(p):
        return np.minimum(tree.query(placed(points, p))[0], .5).mean()

    refined = []
    window = np.array([.65, .65, np.deg2rad(14)])
    for seed in seeds:
        fit = minimize(loss, seed, method='Powell', bounds=list(zip(seed-window, seed+window)),
                       options=dict(xtol=.002, ftol=.0002))
        refined.append((float(fit.fun), fit.x))
    refined.sort(key=lambda row: row[0])
    score, best = refined[0]
    rival = next(((s, p) for s, p in refined if separated(best, p)), None)
    if rival is None:
        raise ValueError('No distinct alternative to assess')
    return best, score, rival


def run(map_file, prefix, inspection):
    started = perf_counter()
    fixed = np.load(map_file)['xyz']
    train = list(scans(prefix))
    if not train:
        raise ValueError('Prefix has no scans with available odometry')
    origin = train[0][2].xyz
    points = endpoints(train, origin)
    best, score, (other_score, other) = search(fixed, points, np.r_[origin[:2], 0.])
    decision_s = perf_counter() - started
    # Freeze the candidate before reading the rest of the inspection.
    later = list(islice((r for r in scans(inspection) if r[0] > train[-1][0]+2), 60))
    if len(later) != 60:
        raise ValueError('Missing independent validation scans')
    check = endpoints(later, origin)
    if not len(check):
        raise ValueError('No elevated endpoints in validation scans')
    tree = cKDTree(fixed)
    def residual(p):
        return float(np.mean(np.minimum(tree.query(placed(check, p))[0], .5)))
    shift = np.r_[best[:2], 0.] - rotate(np.array([origin[0], origin[1], 0.]), best[2])
    identity = np.r_[origin[:2], 0.]
    hold_best, hold_other, hold_identity = residual(best), residual(other), residual(identity)
    unique = other_score - score >= .02
    distinct_on_holdout = hold_other - hold_best >= .02
    decision = 'ambiguous'
    applied = None
    if unique and distinct_on_holdout:
        if hold_best < hold_identity:
            decision = 'candidate_supported'
            applied = [*shift.tolist(), best[2]]
        elif not separated(best, identity):
            decision = 'same_region_keep_odometry'
            applied = [0., 0., 0., 0.]
    return dict(decision=decision, recommended_map_from_odom_xyz_yaw=applied,
                prefix_unique=bool(unique),
                training_scans=len(train), first_header_s=train[0][0], last_header_s=train[-1][0],
                elevated_voxels=len(points), compute_s=decision_s, start_xy_yaw_offset=best.tolist(),
                map_from_odom_xyz=shift.tolist(), map_from_odom_yaw_rad=best[2],
                train_best_m=score, train_rival_m=other_score, rival_xy_yaw_offset=other.tolist(),
                validation_headers_s=[later[0][0], later[-1][0]], validation_scans=len(later),
                validation_voxels=len(check), validation_best_m=hold_best, validation_rival_m=hold_other,
                validation_identity_m=hold_identity)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('map_file', 'prefix', 'inspection'):
        parser.add_argument(name, type=Path)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
