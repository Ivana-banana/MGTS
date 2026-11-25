"""Audit native VLP-16 acquisition times and replay planar gyro deskew for HEAP-1."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation
from numcodecs import blosc

from mgts_nav.bag import reader
from mgts_nav.deskew import deskew_points

DIGEST = "46d8ba658ff7636e2705fc22dd3ed461a06657ac15bd7e88d698fb64631bea74"


def load_arrays(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != DIGEST:
        raise ValueError("Unexpected HEAP-1 raw archive")
    arrays = {}
    with tarfile.open(path) as archive:
        for key in ("points", "time", "timestamp", "valid"):
            prefix = f"velodyne_points/{key}/"
            meta = json.load(archive.extractfile(prefix + ".zarray"))
            chunks = meta["chunks"]
            name = prefix + ".".join("0" for _ in chunks)
            buf = blosc.decompress(archive.extractfile(name).read())
            arrays[key] = np.frombuffer(buf, meta["dtype"]).reshape(chunks)[150:450].copy()
    return arrays


def gyro_poses(data, cfg):
    stamps, rates = [], []
    with reader(data / "mapping") as bag:
        for conn, _, raw in bag.messages(connections=[c for c in bag.connections if c.topic == cfg["imu"]]):
            msg = bag.deserialize(raw, conn.msgtype)
            stamp = msg.header.stamp
            stamps.append(stamp.sec + stamp.nanosec * 1e-9)
            v = msg.angular_velocity
            rates.append((v.x, v.y, v.z))
    clock = np.asarray(stamps)
    if np.any(np.diff(clock) <= 0):
        raise ValueError("Unexpected gyro clock")
    rate = Rotation.from_quat(cfg["base_from_imu_xyzw"]).apply(rates)[:, 2]
    yaw = cumulative_trapezoid(rate, clock, initial=0)
    return np.column_stack((clock, np.zeros((len(clock), 2)), yaw))


def replay(archive, data, output):
    arrays = load_arrays(archive)
    cfg = json.loads((data / "profile.json").read_text())
    poses = gyro_poses(data, cfg)
    mount = Rotation.from_quat(cfg["base_from_lidar_xyzw"])
    lever = np.asarray(cfg["base_from_lidar_xyz_m"])
    store = get_typestore(Stores.ROS2_HUMBLE)
    records, displacements = [], []
    with reader(data / "mapping") as bag, Writer(output, version=8) as writer:
        conns = {c.id: writer.add_connection(c.topic, c.msgtype, typestore=store) for c in bag.connections}
        for c, t, raw in bag.messages():
            if c.topic == cfg["lidar"]:
                i = len(records)
                n = int(arrays["valid"][i, 0])
                xyz = arrays["points"][i, :n]
                dt = arrays["time"][i, :n]
                stamp = float(arrays["timestamp"][i])
                msg = bag.deserialize(raw, c.msgtype)
                if (msg.width != n or msg.data.tobytes() != xyz.tobytes() or
                        t != round(stamp * 1e9) or t != msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec):
                    raise ValueError(f"Unpaired raw cloud at frame {150+i}")
                if not np.all(np.isfinite(dt)):
                    raise ValueError("Invalid point time")
                order = np.argsort(dt, kind="stable")
                az = np.unwrap(np.arctan2(xyz[order, 1], xyz[order, 0]))
                covered = stamp + min(dt.min(), 0) >= poses[0, 0] and stamp + max(dt.max(), 0) <= poses[-1, 0]
                if covered:
                    base = mount.apply(xyz) + lever
                    corrected = deskew_points(base, dt, stamp, poses, cfg["max_sync_gap_s"])
                    if 220 <= 150+i <= 230:
                        displacements.append(np.linalg.norm(corrected-base, axis=1))
                    msg.data = mount.inv().apply(corrected-lever).astype("<f4").view(np.uint8).ravel()
                    raw = store.serialize_cdr(msg, c.msgtype)
                records.append(dict(frame=150+i, count=n, lo=dt.min(), hi=dt.max(),
                                    sweep_deg=np.rad2deg(az[-1]-az[0]), covered=bool(covered)))
            writer.write(conns[c.id], t, raw)
    if len(records) != 300:
        raise ValueError("Expected 300 scans")
    for group in (records, records[70:81]):
        print("frames", group[0]["frame"], group[-1]["frame"], "points", sum(r["count"] for r in group))
        for key in ("lo", "hi", "sweep_deg"):
            values = [r[key] for r in group]
            print(key, np.min(values), np.median(values), np.max(values))
    print("unchanged frames without full IMU coverage", [r["frame"] for r in records if not r["covered"]])
    print("point displacement in metres for 220-230, p50/p95/max", np.percentile(np.concatenate(displacements), [50, 95, 100]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("output_bag", type=Path)
    args = parser.parse_args()
    if args.output_bag.exists():
        parser.error("Output bag already exists")
    with TemporaryDirectory(dir=args.output_bag.parent) as temp:
        stage = Path(temp) / args.output_bag.name
        replay(args.archive, args.data, stage)
        stage.rename(args.output_bag)
