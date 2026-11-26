"""Offline replay of the localizer on a bag, with the same core as the ROS node.

    python3 -m mgts_go2.tools.replay BAG --map DIR --out OUT_DIR [--init odom|search|pose]

Writes OUT_DIR/localization.csv (pose, status, quality and timing per sweep)
and OUT_DIR/summary.json. ``--odom-yaw-drift`` / ``--odom-xy-drift`` corrupt the
odometry on purpose to test that the map keeps the pose; ``--start-offset``
shifts the initial guess.
"""
import argparse
import csv
import json
import sys
import time as _t
from pathlib import Path

import numpy as np

from ..core import bagio, se3
from ..core.deskew import deskew
from ..core.icp import PointMap, residuals_to_map
from ..core.localizer import Localizer, LocalizerParams, MATCHED
from ..core.mapio import read_pcd, read_poses
from ..core.odom_buffer import OdomBuffer


def corrupt_odometry(rows, yaw_rate, xy_rate):
    """Return an OdomBuffer whose poses drift: yaw grows by yaw_rate rad/s, x by xy_rate m/s (in odom)."""
    buf = OdomBuffer(horizon=1e12)
    t0 = rows[0, 0]
    for r in rows:
        dt = r[0] - t0
        t = se3.from_pose(r[1:4], r[4:8])
        drift = se3.make(se3.rot_z(yaw_rate * dt), (xy_rate * dt, 0.0, 0.0))
        buf.add(r[0], drift @ t)
    return buf


def percentiles(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {}
    return {'p50': round(float(np.percentile(a, 50)), 3), 'p95': round(float(np.percentile(a, 95)), 3),
            'max': round(float(a.max()), 3), 'mean': round(float(a.mean()), 3)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag')
    ap.add_argument('--map', required=True, help='directory with map.pcd and mapping_trajectory.csv')
    ap.add_argument('--map-file', default='map.pcd')
    ap.add_argument('--out', required=True)
    ap.add_argument('--init', choices=['odom', 'search', 'pose'], default='odom')
    ap.add_argument('--pose', type=float, nargs=3, metavar=('X', 'Y', 'YAW'))
    ap.add_argument('--start-offset', type=float, nargs=3, default=(0, 0, 0), metavar=('DX', 'DY', 'DYAW_DEG'))
    ap.add_argument('--odom-yaw-drift', type=float, default=0.0, help='deg/s')
    ap.add_argument('--odom-xy-drift', type=float, default=0.0, help='m/s')
    ap.add_argument('--no-prior', action='store_true')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pts, _ = read_pcd(Path(args.map) / args.map_file)
    target = PointMap(pts)
    places = None
    traj = Path(args.map) / 'mapping_trajectory.csv'
    if traj.exists():
        places = read_poses(traj)[1]
    params = LocalizerParams()
    if args.no_prior:
        params.prior_sigma_rot = params.prior_sigma_trans = 0.0
    loc = Localizer(target, params, places)

    odom_true, scans, rows = bagio.load(args.bag, every=args.every, limit=args.limit)
    odom = odom_true
    if args.odom_yaw_drift or args.odom_xy_drift:
        odom = corrupt_odometry(rows, np.radians(args.odom_yaw_drift), args.odom_xy_drift)

    first_t_ob = odom.at(scans[0].stamp)
    offset = se3.make(se3.rot_z(np.radians(args.start_offset[2])), (args.start_offset[0], args.start_offset[1], 0))
    if args.init == 'odom':
        loc.set_map_odom(offset)
    elif args.init == 'pose':
        if not args.pose:
            ap.error('--init pose needs --pose X Y YAW')
        x, y, yaw = args.pose
        p_map = se3.make(se3.rot_z(yaw), (x, y, 0.0))
        loc.set_map_odom(p_map @ se3.inv(se3.planar(first_t_ob)))

    fields = ['stamp', 'status', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'odom_x', 'odom_y', 'odom_yaw',
              'inlier_ratio', 'structure_ratio', 'n_structure', 'search_score', 'rmse', 'n_points', 'iterations', 'degenerate_dims', 'correction_m',
              'correction_rad', 'resid_median', 'resid_within_10cm', 'odom_resid_median', 'odom_resid_within_10cm',
              'deskew_ms', 'prepare_ms', 'icp_ms', 'search_ms', 'total_ms']
    rows_out = []
    t_map_odom_0 = None
    with open(out / 'localization.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in scans:
            t0 = _t.perf_counter()
            try:
                p = deskew(s.points, s.times, s.stamp, odom)
                t_ob = odom.at(s.stamp)
            except LookupError:
                continue
            deskew_ms = (_t.perf_counter() - t0) * 1e3
            r = loc.step(s.stamp, p, t_ob)
            if t_map_odom_0 is None and loc.initialized:
                t_map_odom_0 = r.t_map_odom.copy()
            # independent quality: all points (not the ICP subset) against the map
            res = residuals_to_map(se3.apply(r.t_map_base, p[np.linalg.norm(p, axis=1) > 0.3]), target, 0.3)
            base0 = t_map_odom_0 if t_map_odom_0 is not None else np.eye(4)
            ores = residuals_to_map(se3.apply(base0 @ t_ob, p[np.linalg.norm(p, axis=1) > 0.3]), target, 0.3)
            q = se3.rot_to_quat(r.t_map_base[:3, :3])
            row = {'stamp': '%.9f' % s.stamp, 'status': r.status,
                   'x': r.t_map_base[0, 3], 'y': r.t_map_base[1, 3], 'z': r.t_map_base[2, 3],
                   'qx': q[0], 'qy': q[1], 'qz': q[2], 'qw': q[3],
                   'odom_x': t_ob[0, 3], 'odom_y': t_ob[1, 3], 'odom_yaw': se3.yaw_of(t_ob),
                   'inlier_ratio': r.inlier_ratio, 'structure_ratio': r.structure_ratio,
                   'n_structure': r.n_structure, 'search_score': r.search_score,
                   'rmse': r.rmse, 'n_points': r.n_points,
                   'iterations': r.iterations, 'degenerate_dims': r.degenerate_dims,
                   'correction_m': r.correction_m, 'correction_rad': r.correction_rad,
                   'resid_median': float(np.median(res[np.isfinite(res)])) if np.isfinite(res).any() else float('nan'),
                   'resid_within_10cm': float((res < 0.1).mean()),
                   'odom_resid_median': float(np.median(ores[np.isfinite(ores)])) if np.isfinite(ores).any() else float('nan'),
                   'odom_resid_within_10cm': float((ores < 0.1).mean()),
                   'deskew_ms': deskew_ms}
            for k in ('prepare', 'icp', 'search', 'total'):
                row[k + '_ms'] = r.timings_ms.get(k, float('nan'))
            row['total_ms'] = row['total_ms'] + deskew_ms
            w.writerow(row)
            rows_out.append(row)

    st = [r['status'] for r in rows_out]
    col = lambda k: np.array([float(r[k]) for r in rows_out])
    summary = {
        'bag': args.bag, 'map': str(Path(args.map) / args.map_file), 'init': args.init,
        'start_offset': list(args.start_offset), 'odom_yaw_drift_deg_s': args.odom_yaw_drift,
        'odom_xy_drift_m_s': args.odom_xy_drift, 'prior': not args.no_prior,
        'sweeps': len(rows_out), 'status_counts': {k: st.count(k) for k in sorted(set(st))},
        'matched_ratio': round(st.count(MATCHED) / max(len(st), 1), 4),
        'resid_within_10cm': percentiles(col('resid_within_10cm')),
        'resid_median_m': percentiles(col('resid_median')),
        'odom_resid_within_10cm': percentiles(col('odom_resid_within_10cm')),
        'inlier_ratio': percentiles(col('inlier_ratio')),
        'structure_ratio': percentiles(col('structure_ratio')),
        'correction_m': percentiles(col('correction_m')),
        'timing_ms': {k: percentiles(col(k + '_ms')) for k in ('deskew', 'prepare', 'icp', 'search', 'total')},
        'sweep_period_ms': round(float(np.median(np.diff(col('stamp'))) * 1e3), 1) if len(rows_out) > 1 else None,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
