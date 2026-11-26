"""Check a final ROS 2 result bag against the inspection lidar clock."""
import argparse
import json
from pathlib import Path

import numpy as np

from .bag import cloud_xyz, reader
from .replay_audit import audit as audit_replay


def topic_connections(bag, topic, msgtype):
    selected = [c for c in bag.connections if c.topic == topic]
    if not selected or any(c.msgtype != msgtype for c in selected):
        raise ValueError(f"{topic}: expected {msgtype}")
    return selected


def timestamp_ns(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def audit(inspection, result, scan_topic, map_topic, pose_topic, frame,
          timings=None, allow_missing_initial_scans=0):
    if allow_missing_initial_scans < 0 or (allow_missing_initial_scans and timings is None):
        raise ValueError('Initial scan allowance requires --timings and a nonnegative count')
    with reader(inspection) as bag:
        scans = topic_connections(bag, scan_topic, "sensor_msgs/msg/PointCloud2")
        stamps = [timestamp_ns(bag.deserialize(raw, c.msgtype).header)
                  for c, _, raw in bag.messages(connections=scans)]
    if not stamps or any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError(f"{scan_topic}: expected nonempty, strictly increasing source timestamps")

    source_stamps = set(stamps)
    emitted = set()
    previous = None
    map_messages = 0
    with reader(result) as bag:
        maps = topic_connections(bag, map_topic, "sensor_msgs/msg/PointCloud2")
        poses = topic_connections(bag, pose_topic, "geometry_msgs/msg/PoseStamped")
        for c, _, raw in bag.messages(connections=maps + poses):
            msg = bag.deserialize(raw, c.msgtype)
            if msg.header.frame_id != frame:
                raise ValueError(f"{c.topic}: expected frame {frame}, got {msg.header.frame_id}")
            if c.topic == map_topic:
                xyz = cloud_xyz(msg)
                if not xyz.size or not np.isfinite(xyz).all() or xyz[:, 2].min() == xyz[:, 2].max():
                    raise ValueError(f"{map_topic}: expected finite XYZ points with height variation")
                map_messages += 1
                continue
            t = timestamp_ns(msg.header)
            if t not in source_stamps or (previous is not None and t <= previous):
                raise ValueError(f"{pose_topic}: invalid scan timestamp {t}")
            p, q = msg.pose.position, msg.pose.orientation
            if not np.isfinite([p.x, p.y, p.z]).all():
                raise ValueError(f"{pose_topic}: non-finite position at {t}")
            norm = np.linalg.norm([q.x, q.y, q.z, q.w])
            if not np.isclose(norm, 1, atol=1e-4, rtol=0):
                raise ValueError(f"{pose_topic}: invalid quaternion at {t}")
            emitted.add(t)
            previous = t
    if not emitted or not map_messages:
        raise ValueError("Result requires both a map and poses")
    if timings is not None:
        log = audit_replay(inspection, timings, scan_topic, include_publications=True)
        if (log['missing_headers_s'] or log['extra_headers_s'] or not log['published_poses']
                or log['status']['queue_full'] or log['status']['unprocessed']):
            raise ValueError('Publication log has missing/extra scans, zero poses, or unfinished scans')
        # The existing CSV stores float seconds. Resolve each header to an exact
        # source nanosecond stamp; fail closed if float precision is ambiguous.
        by_seconds = {ns // 10**9 + (ns % 10**9) * 1e-9: ns for ns in stamps}
        if len(by_seconds) != len(stamps):
            raise ValueError('Source timestamps are ambiguous at publication log precision')
        published = {by_seconds[t] for t in log['published_headers_s']}
        unexpected_missing = source_stamps - published - set(stamps[:allow_missing_initial_scans])
        if unexpected_missing:
            raise ValueError(f'Unexpected unpublished scan timestamps: {sorted(unexpected_missing)}')
        if emitted != published:
            raise ValueError(f'Recorded poses differ from publication log: '
                             f'missing={sorted(published - emitted)}, extra={sorted(emitted - published)}')
    return dict(map_messages=map_messages, source_scans=len(stamps), poses=len(emitted),
                coverage=len(emitted) / len(stamps), missing_scan_ns=sorted(source_stamps - emitted))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inspection", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--scan-topic", default="/utlidar/cloud_base")
    parser.add_argument("--map-topic", default="/map_points")
    parser.add_argument("--pose-topic", default="/pose")
    parser.add_argument("--frame", default="map")
    parser.add_argument("--timings", type=Path,
                        help="Enable acceptance: reconcile exact pose stamps with the publication CSV")
    parser.add_argument("--allow-missing-initial-scans", type=int, default=0,
                        help="Allow unpublished poses only among the first N source scans (default: 0)")
    try:
        print(json.dumps(audit(**vars(parser.parse_args()))))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
