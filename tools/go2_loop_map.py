"""3D voxel map of the Go2 mapping run with loop closures on a planar pose graph (#215).

1. Keyframes every KEY_STEP_M or KEY_TURN_RAD of odometry along the mapping run.
2. Odometry edges between consecutive keyframes.
3. Loop edges between keyframes at least MIN_LOOP_S apart and within LOOP_RADIUS_M:
   local submaps are registered from the odometry guess and six perturbed guesses;
   the edge is kept only when fits from clearly different starts agree
   (tools.evaluate_go2_localization.reference_pose).  Its information matrix is the
   planar point-to-plane system at the fit, so a loop inside a straight corridor
   constrains the across-corridor offset and heading but not the along-corridor one.
4. Robust least squares (Huber) over x, y, yaw of all keyframes, first one fixed.
5. Every mapping scan is moved by the planar correction of its nearest keyframe and
   the voxel map is rebuilt; the inspection is then tracked exactly as in localize3d.

`--no-loops` runs the same pipeline without step 3 as a control.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from mgts_nav.localize3d import MAX_MATCH_M, CorrectionFilter, Pose, SurfaceMap, scans, voxel_centroids
from tools.evaluate_go2_localization import (PERTURBATIONS, lift, planar, reference_pose, relative,
                                             submap, wrap)

KEY_STEP_M = 0.5
KEY_TURN_RAD = np.radians(10)
MIN_LOOP_S = 15.0
LOOP_RADIUS_M = 2.5
LOOP_MAX_TURN_RAD = np.radians(120)
VOXEL_M = 0.12
ODOM_SIGMA_M = (0.02, 0.02)      # constant + per metre
ODOM_SIGMA_RAD = (0.005, 0.01)   # constant + per radian
LOOP_SIGMA_M = 0.05
LOOP_EFFECTIVE_POINTS = 50


def keyframes(odom):
    keys = [0]
    for k in range(1, len(odom)):
        rel = relative(odom[keys[-1]], odom[k])
        if np.hypot(*rel[:2]) >= KEY_STEP_M or abs(rel[2]) >= KEY_TURN_RAD:
            keys.append(k)
    return keys


def planar_information(target, source, pose):
    """3x3 information of a planar fit from the point-to-plane system at `pose`."""
    surface = SurfaceMap(target)
    moved = pose.rotation.apply(source)
    world = moved + pose.xyz
    distances, indices = surface.tree.query(world)
    mask = (distances < MAX_MATCH_M) & surface.planar[indices]
    n = surface.normals[indices[mask]]
    p = moved[mask]
    jac = np.column_stack([n[:, 0], n[:, 1], p[:, 0] * n[:, 1] - p[:, 1] * n[:, 0]])
    h = jac.T @ jac / max(1, len(jac)) * LOOP_EFFECTIVE_POINTS / LOOP_SIGMA_M ** 2
    return h + np.eye(3) * 1e-6


def find_loops(times, clouds, odom, planar_odom, keys):
    loops, tried = [], 0
    for b_i, b in enumerate(keys):
        earlier = [a for a in keys[:b_i] if times[b] - times[a] >= MIN_LOOP_S]
        if not earlier:
            continue
        d = [np.hypot(*(planar_odom[a, :2] - planar_odom[b, :2])) for a in earlier]
        a = earlier[int(np.argmin(d))]
        rel = relative(planar_odom[a], planar_odom[b])
        if min(d) > LOOP_RADIUS_M or abs(rel[2]) > LOOP_MAX_TURN_RAD:
            continue
        tried += 1
        target, source = submap(clouds, odom, a), submap(clouds, odom, b)
        guesses = {'odometry': lift(rel, odom[a], odom[b])}
        for k, offset in enumerate(PERTURBATIONS):
            guesses[f'perturbed{k}'] = lift(rel + np.array(offset), odom[a], odom[b])
        ref, agreed, inliers = reference_pose(target, source, guesses)
        if ref is None:
            continue
        info = planar_information(target, source, lift(ref, odom[a], odom[b]))
        values = np.linalg.eigvalsh(info)
        loops.append({'a': a, 'b': b, 'measured': ref.tolist(), 'odometry': rel.tolist(),
                      'inliers': inliers, 'info': info.tolist(),
                      'eigen_ratio': float(values[0] / values[-1])})
    return loops, tried


def optimise(planar_odom, keys, loops):
    index = {k: n for n, k in enumerate(keys)}
    x0 = planar_odom[keys].copy()
    edges = []
    for n in range(1, len(keys)):
        rel = relative(planar_odom[keys[n - 1]], planar_odom[keys[n]])
        s_xy = ODOM_SIGMA_M[0] + ODOM_SIGMA_M[1] * np.hypot(*rel[:2])
        s_th = ODOM_SIGMA_RAD[0] + ODOM_SIGMA_RAD[1] * abs(rel[2])
        edges.append((n - 1, n, rel, np.diag([1 / s_xy, 1 / s_xy, 1 / s_th])))
    for loop in loops:
        root = np.linalg.cholesky(np.array(loop['info'])).T
        edges.append((index[loop['a']], index[loop['b']], np.array(loop['measured']), root))

    def residuals(flat):
        poses = np.vstack([x0[0], flat.reshape(-1, 3)])
        out = []
        for i, j, meas, root in edges:
            delta = relative(poses[i], poses[j]) - meas
            delta[2] = wrap(delta[2])
            out.append(root @ delta)
        return np.concatenate(out)

    sparsity = lil_matrix((3 * len(edges), 3 * (len(keys) - 1)), dtype=int)
    for e, (i, j, _, _) in enumerate(edges):
        for node in (i, j):
            if node:
                sparsity[3 * e:3 * e + 3, 3 * (node - 1):3 * node] = 1
    solution = least_squares(residuals, x0[1:].ravel(), jac_sparsity=sparsity, loss='huber', f_scale=1.0)
    before = residuals(x0[1:].ravel())
    return np.vstack([x0[0], solution.x.reshape(-1, 3)]), {
        'cost_before': float(0.5 * before @ before), 'cost_after': float(2 * solution.cost),
        'nfev': int(solution.nfev), 'status': int(solution.status)}


def corrected_poses(odom, planar_odom, keys, optimised):
    """Planar correction of the nearest keyframe applied to every scan's 3D odometry pose."""
    corrections = []
    for k, opt in zip(keys, optimised):
        raw = planar_odom[k]
        turn = wrap(opt[2] - raw[2])
        c, s = np.cos(turn), np.sin(turn)
        shift = opt[:2] - np.array([c * raw[0] - s * raw[1], s * raw[0] + c * raw[1]])
        corrections.append((turn, shift))
    key_array = np.array(keys)
    poses = []
    for n, pose in enumerate(odom):
        turn, shift = corrections[int(np.argmin(np.abs(key_array - n)))]
        rot = Rotation.from_euler('z', turn)
        poses.append(Pose(rot * pose.rotation, rot.apply(pose.xyz) + np.r_[shift, 0.0]))
    return poses


def track(surface, inspection):
    filt = CorrectionFilter()
    rows = []
    for t, cloud, initial, _ in scans(inspection):
        held = np.arange(len(cloud)) % 4 == 0
        raw, accepted = surface.align(cloud[~held], initial)
        pose = filt.update(t, initial, raw, accepted)
        rows.append([t, *pose.xyz, *pose.rotation.as_quat(), 'matched' if accepted else 'prediction_only',
                     surface.score(cloud[held], pose)])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mapping', type=Path)
    parser.add_argument('inspection', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--no-loops', action='store_true', help='control: same pipeline without loop edges')
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    started = perf_counter()
    stream = list(scans(args.mapping))
    times = np.array([s[0] for s in stream])
    clouds = [s[1] for s in stream]
    odom = [s[2] for s in stream]
    planar_odom = np.array([planar(p.xyz, p.rotation) for p in odom])
    keys = keyframes(planar_odom)
    loops, tried = ([], 0) if args.no_loops else find_loops(times, clouds, odom, planar_odom, keys)
    optimised, solver = optimise(planar_odom, keys, loops)
    poses = corrected_poses(odom, planar_odom, keys, optimised)
    points = voxel_centroids(np.concatenate([p.apply(c) for p, c in zip(poses, clouds)]), VOXEL_M)
    np.savez_compressed(args.out / 'map.npz', xyz=points)
    build_s = perf_counter() - started
    rows = track(SurfaceMap(points), args.inspection)
    with (args.out / 'trajectory.csv').open('w', encoding='utf-8', newline='') as f:
        f.write('t,x,y,z,qx,qy,qz,qw,status,heldout_residual_m\n')
        for row in rows:
            f.write(','.join(str(v) for v in row) + '\n')
    moved = np.hypot(*(optimised[:, :2] - planar_odom[keys][:, :2]).T)
    summary = {
        'mapping_scans': len(stream), 'keyframes': len(keys), 'loop_candidates': tried,
        'loop_edges': len(loops), 'loop_eigen_ratio_p50': float(np.median([l['eigen_ratio'] for l in loops])) if loops else None,
        'solver': solver, 'keyframe_shift_m_p50_p95_max': np.round(np.percentile(moved, [50, 95, 100]), 4).tolist(),
        'map_voxels': len(points), 'build_s': round(build_s, 1),
        'inspection_scans': len(rows), 'matched': sum(r[8] == 'matched' for r in rows),
        'heldout_mean_residual_m': float(np.mean([r[9] for r in rows])),
    }
    (args.out / 'loops.json').write_text(json.dumps(loops, indent=1) + '\n', encoding='utf-8')
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
