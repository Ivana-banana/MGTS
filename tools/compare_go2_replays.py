"""Compare poses, statuses and playback clock boundaries of two live Go2 runs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from mgts_nav.bag import reader
from mgts_nav.result_bag_audit import timestamp_ns, topic_connections


def read_run(path):
    poses = {}
    with reader(path / "result") as bag:
        cs = topic_connections(bag, "/mgts/pose3d", "geometry_msgs/msg/PoseStamped")
        for conn, _, raw in bag.messages(connections=cs):
            msg = bag.deserialize(raw, conn.msgtype)
            p, q = msg.pose.position, msg.pose.orientation
            t = timestamp_ns(msg.header)
            if t in poses:
                raise ValueError(f"{path}: duplicate pose at {t}")
            poses[t] = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
    with (path / "arrivals.csv").open(newline="") as stream:
        clock = {int(r["header_ns"]): int(r["last_replay_clock_ns"]) for r in csv.DictReader(stream)}
    if not poses or clock.keys() != poses.keys() or any(v <= 0 for v in clock.values()):
        raise ValueError(f"{path}: missing poses or replay clock")
    with (path / "timings.csv").open(newline="") as stream:
        status = {r["header_s"]: r["status"] for r in csv.DictReader(stream)}
    return poses, clock, status


def compare(prefix, full, cutoff):
    short, short_clock, short_status = read_run(prefix)
    long, long_clock, long_status = read_run(full)
    common = sorted(short.keys() & long.keys())
    if not common:
        raise ValueError("No common pose stamps")
    a = np.array([short[t] for t in common])
    b = np.array([long[t] for t in common])
    rotation_delta = Rotation.from_quat(a[:, 3:]).inv() * Rotation.from_quat(b[:, 3:])
    before = {t for t, clock in short_clock.items() if clock < cutoff}
    before_full = {t for t, clock in long_clock.items() if clock < cutoff}
    return dict(prefix_poses=len(short), full_poses=len(long), common=len(common),
                prefix_only=len(short) - len(common),
                observed_before_both=len(before & before_full),
                full_before_missing_in_prefix=len(before_full - short.keys()),
                prefix_status_mismatches=sum(long_status.get(t) != s for t, s in short_status.items()),
                max_position_difference_m=float(np.linalg.norm(a[:, :3] - b[:, :3], axis=1).max()),
                max_angle_difference_deg=float(np.rad2deg(rotation_delta.magnitude()).max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("full", type=Path)
    parser.add_argument("cutoff", type=int)
    print(json.dumps(compare(**vars(parser.parse_args()))))
