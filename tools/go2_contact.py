"""Compare Go2 foot-load measurements with motion in one-second recording-time windows."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

from mgts_nav.bag import reader, stamp


def analyze(path, trajectory=None):
    low, odom, scans = [], [], []
    with reader(path) as bag:
        selected = [c for c in bag.connections if c.topic in
                    ('/lowstate', '/utlidar/robot_odom', '/utlidar/cloud_base')]
        for c, ns, raw in bag.messages(connections=selected):
            msg = bag.deserialize(raw, c.msgtype)
            t = ns / 1e9
            if c.topic == '/lowstate':
                low.append([t, msg.tick, *msg.foot_force, *msg.foot_force_est])
            elif c.topic == '/utlidar/cloud_base':
                scans.append([stamp(msg), t])
            else:
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                odom.append([t, stamp(msg), p.x, p.y, q.x, q.y, q.z, q.w])
    low, odom, scans = map(np.array, (low, odom, scans))
    force = low[:, 2:6]
    tick = low[:, 1]
    elapsed = low[:, 0] - low[0, 0]
    tick_error = elapsed - (tick - tick[0]) / 1000
    report = dict(samples=len(low), foot_min=force.min(axis=0).tolist(),
                  foot_median=np.median(force, axis=0).tolist(), foot_max=force.max(axis=0).tolist(),
                  estimated_range=[low[:, 6:].min(), low[:, 6:].max()],
                  tick_step_ms=np.percentile(np.diff(tick), [0, 50, 100]).tolist(),
                  tick_delivery_drift_ms=np.percentile(tick_error * 1000, [0, 50, 100]).tolist(),
                  odom_delivery_minus_header_s=np.percentile(odom[:, 0]-odom[:, 1], [1, 50, 99]).tolist(),
                  lidar_delivery_minus_header_s=np.percentile(scans[:, 1]-scans[:, 0], [1, 50, 99]).tolist())
    fits = {}
    if trajectory:
        with trajectory.open(newline='') as stream:
            fits = {float(r['t']): (float(r['fit_residual_m']), float(r['odom_residual_m'])) for r in csv.DictReader(stream)
                    if r['status'] == 'matched'}
    scan_fit = np.array([fits.get(t, (np.nan, np.nan)) for t in scans[:, 0]])
    # Express lidar header times on the delivery clock using nearby odometry.
    scan_wall = np.interp(scans[:, 0], odom[:, 1], odom[:, 0], left=np.nan, right=np.nan)
    yaw = np.unwrap(Rotation.from_quat(odom[:, 4:8]).as_euler('xyz')[:, 2])
    rows = []
    for sec in range(int(elapsed[-1])):
        start = low[0, 0] + sec
        selected = (elapsed >= sec) & (elapsed < sec + 1)
        idx = np.flatnonzero((odom[:, 0] >= start) & (odom[:, 0] < start+1))
        if selected.sum() < 2 or len(idx) < 2:
            continue
        feet = force[selected]
        contact = feet >= 20
        speed = np.linalg.norm(np.diff(odom[idx, 2:4], axis=0), axis=1).sum()
        turn = abs(yaw[idx[-1]] - yaw[idx[0]])
        f = scan_fit[(scan_wall >= start) & (scan_wall < start+1) & np.isfinite(scan_fit).all(axis=1)]
        rows.append(dict(start_s=sec, motion_m=float(speed), turn_rad=float(turn),
                         changes=int(np.count_nonzero(np.diff(contact, axis=0))),
                         mean_loaded=float(contact.sum(axis=1).mean()),
                         fit_m=float(np.median(f[:, 0])) if len(f) else None,
                         odom_m=float(np.median(f[:, 1])) if len(f) else None))
    report['quiet'] = min(rows, key=lambda r: r['motion_m'] + r['turn_rad'])
    report['turn'] = max(rows, key=lambda r: r['turn_rad'])
    valid = [r for r in rows if r['fit_m'] is not None]
    if valid:
        report['windows_with_fit'] = len(valid)
        for key in ('fit_m', 'odom_m'):
            corr = spearmanr([r['changes'] for r in valid], [r[key] for r in valid])
            report[f'changes_{key}_spearman'] = float(corr.statistic)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--trajectory', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.recording, args.trajectory), indent=2))
