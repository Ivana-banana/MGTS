"""Audit subscriber callback timestamps against the inspection and both result bags."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from mgts_nav.bag import reader
from mgts_nav.result_bag_audit import timestamp_ns, topic_connections


def stamps(path, topic, msgtype):
    with reader(path) as bag:
        conns = topic_connections(bag, topic, msgtype)
        return Counter(timestamp_ns(bag.deserialize(raw, c.msgtype).header)
                       for c, _, raw in bag.messages(connections=conns))


def distribution(values):
    if not values:
        return None
    return dict(zip(('p50', 'p95', 'p99', 'max'), np.percentile(values, [50, 95, 99, 100]).tolist()))


def callback_times(folder):
    with (folder / 'timings.csv').open(newline='') as stream:
        return distribution([float(r['latency_ms']) for r in csv.DictReader(stream)
                             if r['status'] in ('matched', 'prediction_only')])


def audit(inspection, measurement, reference):
    seen = {'scan': {}, 'pose': {}}
    duplicates = Counter()
    with (measurement/'observations.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            topic = row['topic']
            ns = int(row['sec']) * 10**9 + int(row['nanosec'])
            if ns in seen[topic]:
                duplicates[topic] += 1
            else:
                seen[topic][ns] = int(row['callback_monotonic_ns'])
    src = stamps(inspection, '/utlidar/cloud_base', 'sensor_msgs/msg/PointCloud2')
    current = stamps(measurement/'replay/result', '/mgts/pose3d', 'geometry_msgs/msg/PoseStamped')
    released = stamps(reference/'result', '/mgts/pose3d', 'geometry_msgs/msg/PoseStamped')
    scan, pose = seen['scan'], seen['pose']
    common = sorted(scan.keys() & pose.keys())
    delta = [(pose[t] - scan[t]) / 1e6 for t in common]
    return dict(source_scans=src.total(), captured_poses=current.total(), released_poses=released.total(),
                observed_scans=len(scan), observed_poses=len(pose), matched=len(common),
                duplicate_observations=dict(duplicates),
                duplicate_bag_stamps=dict(scan=src.total()-len(src), pose=current.total()-len(current)),
                unseen_scans=sorted(src.keys()-scan.keys()), unseen_poses=sorted(current.keys()-pose.keys()),
                foreign_scans=sorted(scan.keys()-src.keys()), foreign_poses=sorted(pose.keys()-current.keys()),
                unpaired_scan_stamps=sorted(scan.keys()-pose.keys()), unpaired_pose_stamps=sorted(pose.keys()-scan.keys()),
                changed_release_stamps=current != released, negative=sum(v < 0 for v in delta),
                over_71_99_ms=sum(v > 71.99 for v in delta), subscriber_delta_ms=distribution(delta),
                repeat_callback_ms=callback_times(measurement/'replay'), release_callback_ms=callback_times(reference))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inspection', 'measurement', 'reference'):
        parser.add_argument(name, type=Path)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))
