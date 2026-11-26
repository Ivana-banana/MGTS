"""Residuals of the published 3D track on sharp turns and accelerations (#207).

Episodes are selected from the stock odometry only, before any residual is read:

* yaw rate and planar speed are central differences of odometry over +-0.1 s,
  the speed change rate over +-0.15 s;
* a turn episode is a contiguous run with |yaw rate| >= TURN_EDGE_DEG_S whose peak
  reaches TURN_PEAK_DEG_S; the TOP_N runs with the highest peaks are kept;
* an acceleration episode is a contiguous run with |d speed/dt| >= ACCEL_EDGE whose
  peak reaches ACCEL_PEAK while |yaw rate| < ACCEL_MAX_TURN_DEG_S at the peak and
  that does not touch a turn episode; per direction the TOP_N highest peaks are kept;
* runs shorter than MIN_SCANS scans are ignored;
* control scans are steady straight walking outside every episode +-GUARD_S.

For each episode the published poses (/mgts/pose3d) and the odometry are compared on
the same scan stamps by the median distance from held-out points (every fourth point,
excluded from the 3D fit) to the fixed map.  Without an external reference this is a
consistency measure with the map, not a position error.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from mgts_nav.bag import reader, stamp
from mgts_nav.localize3d import Pose, scans

TURN_EDGE_DEG_S = 45.0
TURN_PEAK_DEG_S = 90.0
ACCEL_EDGE = 0.6
ACCEL_PEAK = 1.2
ACCEL_MAX_TURN_DEG_S = 30.0
CONTROL_MAX_TURN_DEG_S = 10.0
CONTROL_MAX_ACCEL = 0.3
CONTROL_MIN_SPEED = 0.2
GUARD_S = 0.5
TOP_N = 2  # per direction: left/right turns, speed-up/braking
MIN_SCANS = 3


def derivative(values, times, half):
    out = np.full(len(times), np.nan)
    lo = np.searchsorted(times, times - half)
    hi = np.searchsorted(times, times + half, side='right') - 1
    ok = hi > lo
    out[ok] = (values[hi[ok]] - values[lo[ok]]) / (times[hi[ok]] - times[lo[ok]])
    return out


def motion(times, xy, yaw):
    rate = np.degrees(derivative(np.unwrap(yaw), times, 0.1))
    speed = np.hypot(derivative(xy[:, 0], times, 0.1), derivative(xy[:, 1], times, 0.1))
    accel = derivative(speed, times, 0.15)
    return rate, speed, accel


def runs(mask):
    """Inclusive [start, end] index ranges of True runs."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2] - 1))


def pick(signal, edge, peak, allowed=None, top=TOP_N):
    signal = np.nan_to_num(signal)
    chosen = []
    for sign in (1, -1):
        directed = []
        for a, b in runs(sign * signal >= edge):
            k = a + int(np.argmax(sign * signal[a:b + 1]))
            if b - a + 1 >= MIN_SCANS and sign * signal[k] >= peak and (allowed is None or allowed(a, b, k)):
                directed.append((float(sign * signal[k]), a, b, k))
        chosen += sorted(directed, reverse=True)[:top]
    return sorted((a, b, k) for _, a, b, k in chosen)


def select(times, xy, yaw):
    rate, speed, accel = motion(times, xy, yaw)
    turns = pick(rate, TURN_EDGE_DEG_S, TURN_PEAK_DEG_S)

    def near(a, b, episodes):
        return any(times[a] - GUARD_S <= times[e] and times[s] <= times[b] + GUARD_S for s, e, _ in episodes)

    accels = pick(accel, ACCEL_EDGE, ACCEL_PEAK,
                  lambda a, b, k: abs(rate[k]) < ACCEL_MAX_TURN_DEG_S and not near(a, b, turns))
    t0 = times[0]
    episodes = []
    for kind, found in (('turn', turns), ('acceleration', accels)):
        for a, b, k in found:
            if kind == 'turn':
                angle = float(np.degrees(np.unwrap(yaw)[b] - np.unwrap(yaw)[a]))
                detail = {'peak_deg_s': round(float(rate[k]), 1), 'turn_deg': round(angle, 1),
                          'direction': 'left' if rate[k] > 0 else 'right'}
            else:
                detail = {'peak_m_s2': round(float(accel[k]), 2),
                          'speed_m_s': [round(float(speed[a]), 2), round(float(speed[b]), 2)],
                          'direction': 'speed-up' if accel[k] > 0 else 'braking'}
            episodes.append({'kind': kind, 'start_s': round(float(times[a] - t0), 3),
                             'end_s': round(float(times[b] - t0), 3), 'scans': list(range(a, b + 1)), **detail})
    taken = np.zeros(len(times), bool)
    for e in episodes:
        taken |= (times >= times[e['scans'][0]] - GUARD_S) & (times <= times[e['scans'][-1]] + GUARD_S)
    control = (~taken & (np.abs(rate) < CONTROL_MAX_TURN_DEG_S) & (np.abs(accel) < CONTROL_MAX_ACCEL)
               & (speed >= CONTROL_MIN_SPEED))
    return episodes, np.flatnonzero(control)


def published(path):
    poses = {}
    with reader(path) as bag:
        for c, _, raw in bag.messages(connections=[c for c in bag.connections if c.topic == '/mgts/pose3d']):
            msg = bag.deserialize(raw, c.msgtype)
            p, q = msg.pose.position, msg.pose.orientation
            key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            poses[key] = Pose(Rotation.from_quat([q.x, q.y, q.z, q.w]), np.array([p.x, p.y, p.z]))
    return poses


def summary(indices, residuals):
    out = {'scans': len(indices)}
    for name, values in residuals.items():
        got = np.array([values[i] for i in indices if np.isfinite(values[i])])
        out[name] = {'poses': int(len(got)), 'coverage': round(len(got) / max(1, len(indices)), 4),
                     'median_cm': round(float(np.median(got)) * 100, 2) if len(got) else None,
                     'p95_cm': round(float(np.percentile(got, 95)) * 100, 2) if len(got) else None}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inspection', type=Path)
    parser.add_argument('--result', type=Path, help='published rosbag2 with /mgts/pose3d')
    parser.add_argument('--map', type=Path, help='fixed map.npz used by the published run')
    parser.add_argument('--select-only', action='store_true', help='print episodes, read no residuals')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    stream = list(scans(args.inspection))
    times = np.array([s[0] for s in stream])
    odom = [s[2] for s in stream]
    xy = np.array([p.xyz[:2] for p in odom])
    yaw = np.array([p.rotation.as_euler('ZYX')[0] for p in odom])
    episodes, control = select(times, xy, yaw)
    rule = {k: v for k, v in globals().items() if k.isupper()}
    if args.select_only:
        print(json.dumps({'rule': rule, 'control_scans': len(control),
                          'episodes': [{k: v for k, v in e.items() if k != 'scans'} | {'n': len(e['scans'])}
                                       for e in episodes]}, indent=2, ensure_ascii=False))
        return
    if not (args.result and args.map and args.out):
        parser.error('--result, --map and --out are required unless --select-only')
    poses = published(args.result)
    with np.load(args.map) as data:
        tree = cKDTree(data['xyz'])
    residuals = {'odometry': np.full(len(times), np.nan), '3d': np.full(len(times), np.nan)}
    with reader(args.inspection) as bag:
        keys = []
        lidar = [c for c in bag.connections if c.topic == '/utlidar/cloud_base']
        for c, _, raw in bag.messages(connections=lidar):
            h = bag.deserialize(raw, c.msgtype).header.stamp
            keys.append((stamp(bag.deserialize(raw, c.msgtype)), (h.sec, h.nanosec)))
    key_of = {round(t, 6): k for t, k in keys}
    for i, (t, cloud, pose, _) in enumerate(stream):
        held = cloud[np.arange(len(cloud)) % 4 == 0]
        residuals['odometry'][i] = np.median(tree.query(pose.apply(held))[0])
        live = poses.get(key_of.get(round(t, 6)))
        if live is not None:
            residuals['3d'][i] = np.median(tree.query(live.apply(held))[0])
    report = {'rule': rule, 'inputs': {'inspection': str(args.inspection), 'result': str(args.result),
                                       'map': str(args.map)},
              'all': summary(range(len(times)), residuals), 'control': summary(control, residuals), 'episodes': []}
    for e in episodes:
        report['episodes'].append({k: v for k, v in e.items() if k != 'scans'} | summary(e['scans'], residuals))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (args.out / 'residuals.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'odometry_m', '3d_m'])
        writer.writerows(zip(times, residuals['odometry'], residuals['3d']))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
