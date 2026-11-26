"""Summarise live 3D-node runs on copies with odometry removed (#209).

Expects ROOT/control (the unmodified window made by tools.drop_odometry) and, for
each variant, ROOT/<name>-run with timings.csv, arrivals.csv and result/ written by
tools.go2_live_replay.  Scan identity is the integer header stamp of cloud_base;
delivery times come from the control copy.  The recovered pose is the first
published pose whose header is not earlier than the first odometry header kept
after the removed interval.  Pose differences are taken against the control run on
the same headers.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from mgts_nav.bag import reader

CLOUD = '/utlidar/cloud_base'
ODOM = '/utlidar/robot_odom'


def header_ns(message):
    return message.header.stamp.sec * 10 ** 9 + message.header.stamp.nanosec


def stamps(bag_path):
    """(delivery_ns, header_ns) lists for cloud_base and robot_odom."""
    out = {CLOUD: [], ODOM: []}
    with reader(bag_path) as bag:
        conns = [c for c in bag.connections if c.topic in out]
        for c, t, raw in bag.messages(connections=conns):
            out[c.topic].append((t, header_ns(bag.deserialize(raw, c.msgtype))))
    return out


def load(run, delivery):
    with (run / 'timings.csv').open(encoding='utf-8') as f:
        timing = list(csv.DictReader(f))
    with (run / 'arrivals.csv').open(encoding='utf-8') as f:
        arrivals = {int(r['header_ns']): int(r['last_replay_clock_ns']) for r in csv.DictReader(f)}
    poses = {}
    with reader(run / 'result') as bag:
        for c, _, raw in bag.messages(connections=[c for c in bag.connections if c.topic == '/mgts/pose3d']):
            m = bag.deserialize(raw, c.msgtype)
            p, q = m.pose.position, m.pose.orientation
            poses[header_ns(m)] = (np.array([p.x, p.y, p.z]), Rotation.from_quat([q.x, q.y, q.z, q.w]))
    skipped = []
    for r in timing:
        if r['status'] not in ('matched', 'prediction_only'):
            h = min(delivery, key=lambda k: abs(k / 1e9 - float(r['header_s'])))
            skipped.append({'header_ns': h, 'delivery_ns': delivery[h], 'status': r['status']})
    # The CSV logs the queue after popleft; queue_full briefly holds one more scan.
    peak = max(int(r['pending']) + 1 for r in timing)
    stats = {'rows': len(timing), 'poses': len(poses), 'status': dict(Counter(r['status'] for r in timing)),
             'max_pending_csv': max(int(r['pending']) for r in timing), 'peak_before_removal': peak,
             'max_retained': min(8, peak), 'skipped': skipped}
    return stats, poses, arrivals


def report(root, drop_start_ns, variants):
    source = stamps(root / 'control')
    delivery = {h: t for t, h in source[CLOUD]}
    control, base, _ = load(root / 'control-run', delivery)
    results = {'control': control}
    for name, duration_ns in variants.items():
        stats, poses, arrivals = load(root / f'{name}-run', delivery)
        end = drop_start_ns + duration_ns
        after = next(row for row in source[ODOM] if row[0] >= end)
        h = min(k for k in poses if k >= after[1])
        prev = max(k for k in poses if delivery[k] < drop_start_ns)
        p, q = poses[h]
        b, bq = base[h]
        pp, pq = poses[prev]
        bp, bpq = base[prev]
        stats.update(
            recovery_header_ns=h, recovery_cloud_delivery_ns=delivery[h], recovery_arrival_clock_ns=arrivals[h],
            sensor_delay_s=(h - after[1]) / 1e9, input_delivery_delay_s=(delivery[h] - end) / 1e9,
            arrival_clock_delay_s=(arrivals[h] - end) / 1e9,
            position_difference_m=float(np.linalg.norm(p - b)), angle_difference_rad=float((q * bq.inv()).magnitude()),
            jump_m=float(np.linalg.norm(p - pp)), control_same_endpoints_jump_m=float(np.linalg.norm(b - bp)),
            excess_jump_vector_m=float(np.linalg.norm((p - pp) - (b - bp))),
            jump_angle_rad=float((q * pq.inv()).magnitude()), control_jump_angle_rad=float((bq * bpq.inv()).magnitude()),
            pre_gap_header_ns=prev)
        results[name] = stats
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', type=Path)
    parser.add_argument('--drop-start-ns', type=int, required=True)
    parser.add_argument('--variant', action='append', default=[], metavar='NAME=DURATION_NS',
                        help='removed interval length per variant, e.g. drop03=300000000')
    args = parser.parse_args()
    variants = {k: int(v) for k, v in (item.split('=', 1) for item in args.variant)}
    results = report(args.root, args.drop_start_ns, variants)
    text = json.dumps(results, indent=2)
    (args.root / 'metrics.json').write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
