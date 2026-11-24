"""Estimate a constant header-clock lag of odometry or IMU against the lidar.

    python tools/clock_lag.py BAG --config PROFILE.json --trajectory OUT/map/trajectory.csv

The yaw rate of lidar-matched poses (header time of the scans) is compared with the
yaw rate of odometry, or with the IMU gyro in the base frame, for lags of +-1 s.
`header_offset_s` is the value to put in the profile: it already has the opposite sign of
the measured lag and includes any offset the profile applies now. Needs turns in the
recording; a flat correlation peak (low `peak_correlation`) means "not measurable".
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mgts_nav.bag import load_config, reader, stamp  # noqa: E402


def sensor_rate(bag_path, cfg, role):
    rows = []
    offset = cfg.get('header_offset_s', {}).get(role, 0.0)
    with reader(bag_path) as bag:
        conns = [c for c in bag.connections if c.topic == cfg[role]]
        for conn, _, raw in bag.messages(connections=conns):
            m = bag.deserialize(raw, conn.msgtype)
            if role == 'odom':
                q = m.pose.pose.orientation
                rows.append([stamp(m) + offset, Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('zyx')[0]])
            else:
                w = m.angular_velocity
                rows.append([stamp(m) + offset,
                             Rotation.from_quat(cfg['base_from_imu_xyzw']).apply([w.x, w.y, w.z])[2]])
    a = np.asarray(rows)
    if role == 'odom':
        yaw = np.unwrap(a[:, 1])
        return a[1:, 0] - np.diff(a[:, 0]) / 2, np.diff(yaw) / np.diff(a[:, 0])
    return a[:, 0], a[:, 1]


def trajectory_rate(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = [(float(r['t']), float(r['yaw'])) for r in csv.DictReader(f)
                if r.get('status', 'matched') != 'prediction_only']
    a = np.asarray(rows)
    yaw = np.unwrap(a[:, 1])
    return a[1:, 0] - np.diff(a[:, 0]) / 2, np.diff(yaw) / np.diff(a[:, 0])


def estimate(ref_t, ref_rate, t, rate, max_lag=1.0, step=0.005, max_hole=0.15):
    lags = np.arange(-max_lag, max_lag + step / 2, step)
    grid = np.arange(ref_t[0], ref_t[-1], 0.01)
    reference = np.interp(grid, ref_t, ref_rate)
    scores = []
    for lag in lags:
        probe = grid + lag
        index = np.clip(np.searchsorted(t, probe), 1, len(t) - 1)
        hole = np.minimum(np.abs(t[index] - probe), np.abs(t[index - 1] - probe)) > max_hole
        ok = ~hole & (probe >= t[0]) & (probe <= t[-1])
        if ok.sum() < 0.5 * len(grid):
            scores.append(np.nan)
            continue
        a, b = reference[ok], np.interp(probe[ok], t, rate)
        a, b = a - a.mean(), b - b.mean()
        den = np.sqrt((a @ a) * (b @ b))
        scores.append(a @ b / den if den > 0 else np.nan)
    scores = np.asarray(scores)
    best = int(np.nanargmax(scores))
    return float(lags[best]), float(scores[best])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--trajectory', required=True, type=Path, help='map/trajectory.csv of this bag')
    parser.add_argument('--role', choices=['odom', 'imu'], default='odom')
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        if cfg[args.role] is None:
            parser.exit(2, f'{args.role} is disabled in the profile\n')
        lag, score = estimate(*trajectory_rate(args.trajectory), *sensor_rate(args.bag, cfg, args.role))
    except (OSError, ValueError) as exc:
        parser.exit(2, f'{exc}\n')
    current = cfg.get('header_offset_s', {}).get(args.role, 0.0)
    print(json.dumps(dict(role=args.role, lag_s=lag, peak_correlation=score,
                          header_offset_s={args.role: round(current - lag, 3)}), indent=2))


if __name__ == '__main__':
    main()
