"""Compare the two official Go2 cloud frames using shared intensity sequences."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from mgts_nav.bag import cloud_field, cloud_xyz, reader, stamp


def inspect(path):
    base = []
    deskewed = []
    odometry = []
    padding = Counter()
    with reader(path) as bag:
        wanted = {"/utlidar/cloud_base", "/utlidar/cloud_deskewed", "/utlidar/robot_odom"}
        conns = [c for c in bag.connections if c.topic in wanted]
        for c, _, raw in bag.messages(connections=conns):
            m = bag.deserialize(raw, c.msgtype)
            t = stamp(m)
            if c.topic.endswith("robot_odom"):
                p, q = m.pose.pose.position, m.pose.pose.orientation
                odometry.append((t, [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]))
                continue
            xyz = cloud_xyz(m)
            valid = np.isfinite(xyz).all(axis=1) & (xyz != 0).any(axis=1)
            intensity = cloud_field(m, "intensity", {7: "f4"})[valid]
            signature = hashlib.sha256(intensity.tobytes()).digest()
            row = (t, signature, xyz[valid])
            if c.topic.endswith("deskewed"):
                padding[int((xyz == 0).all(axis=1).sum())] += 1
                deskewed.append(row)
            else:
                base.append(row)
    if not base or not deskewed or len(odometry) < 2:
        raise ValueError(f"{path}: missing required Go2 data")
    times, positions, quats = zip(*odometry)
    times, positions = np.asarray(times), np.asarray(positions)
    rotations = Slerp(times, Rotation.from_quat(quats))
    by_signature = defaultdict(list)
    for t, signature, points in deskewed:
        by_signature[signature].append((t, points))
    deskewed_times = np.array([r[0] for r in deskewed])
    rows = []
    offsets = []
    nearest_agrees = 0
    outside = 0
    for t, signature, points in base:
        candidates = [(s, target) for s, target in by_signature[signature] if abs(s - t) < .20]
        if len(candidates) != 1:
            continue
        s, target = candidates[0]
        offsets.append((s - t) * 1000)
        nearest_agrees += int(deskewed_times[np.abs(deskewed_times - t).argmin()] == s)
        if not times[0] <= t <= times[-1]:
            outside += 1
            continue
        pos = [np.interp(t, times, axis) for axis in positions.T]
        predicted = rotations(t).apply(points) + pos
        start, end = max(times[0], t - .1), min(times[-1], t + .1)
        rate = (rotations(start).inv() * rotations(end)).magnitude() / (end - start)
        tree = cKDTree(target)
        transformed = np.median(tree.query(predicted)[0])
        identity = np.median(tree.query(points)[0])
        rows.append((np.degrees(rate), transformed, identity))
    if not rows:
        raise ValueError(f"{path}: no matched scans with odometry")
    rows = np.array(rows)
    groups = {}
    for key, mask in (("turn_ge_20_deg_s", rows[:, 0] >= 20), ("other", rows[:, 0] < 20)):
        selected = rows[mask]
        groups[key] = {"scans": len(selected)}
        if len(selected):
            for i, method in ((1, "odom"), (2, "identity")):
                groups[key][method + "_median_m"] = float(np.median(selected[:, i]))
                groups[key][method + "_p95_m"] = float(np.percentile(selected[:, i], 95))
    return dict(base=len(base), deskewed=len(deskewed), uniquely_matched=len(offsets),
                closest_header_same_scan=nearest_agrees, outside_odom=outside,
                zero_points_per_deskewed=dict(padding),
                header_delay_ms_p05_p50_p95=np.percentile(offsets, [5, 50, 95]).tolist(),
                distances=groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps({str(path): inspect(path) for path in args.bags}, indent=2))
