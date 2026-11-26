"""Reconcile the input lidar bag with the ROS localizer's timing log."""

import argparse
import csv
import json
from collections import Counter
from math import isfinite
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReaderError
from rosbags.serde.errors import SerdeError

from .bag import reader, stamp


def scan_headers(path, topic):
    with reader(path) as bag:
        connections = [conn for conn in bag.connections if conn.topic == topic
                       and conn.msgtype == 'sensor_msgs/msg/PointCloud2']
        if not connections:
            raise ValueError(f'{path}: no PointCloud2 topic {topic}')
        headers = Counter(stamp(bag.deserialize(raw, conn.msgtype))
                          for conn, _, raw in bag.messages(connections=connections))
    if not headers:
        raise ValueError(f'{path}: no scans on {topic}')
    return headers


def audit(source, timings, topic):
    expected = scan_headers(source, topic)
    logged = Counter()
    statuses = Counter()
    pose_latencies = []
    max_pending = 0
    with timings.open(encoding='utf-8-sig', newline='') as stream:
        rows = csv.DictReader(stream)
        fields = ('header_s', 'status', 'latency_ms', 'pending')
        if not set(fields).issubset(rows.fieldnames or []):
            raise ValueError(f'{timings}: required columns are {",".join(fields)}')
        for line, row in enumerate(rows, 2):
            try:
                header = float(row['header_s'])
                latency = float(row['latency_ms'])
                pending = int(row['pending'])
                status = row['status']
                if not isfinite(header) or not isfinite(latency) or latency < 0 or pending < 0 or not status:
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise ValueError(f'{timings}:{line}: invalid timing row') from error
            logged[header] += 1
            statuses[status] += 1
            max_pending = max(max_pending, pending)
            if status in ('matched', 'prediction_only'):
                pose_latencies.append(latency)

    percentiles = None
    if pose_latencies:
        p50, p95, p99 = np.percentile(pose_latencies, (50, 95, 99)).tolist()
        percentiles = dict(p50=p50, p95=p95, p99=p99)
    return dict(input_scans=sum(expected.values()), logged_scans=sum(logged.values()),
                missing_headers_s=sorted((expected - logged).elements()),
                extra_headers_s=sorted((logged - expected).elements()), status=statuses,
                max_pending=max_pending, pose_latency_ms=percentiles)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='Exit 1 for missing or extra scans, queue overflow, or scans pending at shutdown.')
    parser.add_argument('source', type=Path, help='Full input bag used for replay')
    parser.add_argument('timings', type=Path, help='timings.csv from ros_localize3d')
    parser.add_argument('--lidar', default='/utlidar/cloud_base')
    args = parser.parse_args()
    try:
        result = audit(args.source, args.timings, args.lidar)
    except (AnyReaderError, SerdeError, OSError, ValueError) as error:
        parser.exit(2, f'{error}\n')
    print(json.dumps(result, indent=2))
    status = result['status']
    return int(bool(result['missing_headers_s'] or result['extra_headers_s']
                    or status['queue_full'] or status['unprocessed']))


if __name__ == '__main__':
    raise SystemExit(main())
