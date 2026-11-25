"""Convert fixed, disjoint Intel Research Lab FLASER windows to lidar-only bags."""
import argparse
import csv
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from mgts_nav.slam import relative

HASHES = {
    "intel.log.gz": "449a9813e9709b6fd053ba0d006ea69c5f4485ba08dd021448e653e3af2e6ff9",
    "intel.gfs.log.gz": "152d4db32761a71ac4a557d2bd383d4ba054864c290b799df92096ba34b70dea",
}


def laser_rows(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[path.name]:
        raise ValueError(f"Unexpected digest for {path.name}")
    with gzip.open(path, "rt", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("FLASER "):
                fields = line.split()
                if len(fields) != 191 or fields[1] != "180":
                    raise ValueError("Unexpected FLASER layout")
                yield fields


def points(ranges):
    angles = np.deg2rad(np.arange(180) - 90)
    valid = np.isfinite(ranges) & (ranges > 0) & (ranges < 20)
    return np.column_stack((ranges[valid]*np.cos(angles[valid]), ranges[valid]*np.sin(angles[valid]),
                            np.zeros(valid.sum()))).astype("<f4")


def write_window(path, rows):
    store = get_typestore(Stores.ROS2_HUMBLE)
    types = store.types
    field = types["sensor_msgs/msg/PointField"]
    fields = [field(name, i*4, 7, 1) for i, name in enumerate("xyz")]
    with Writer(path, version=8) as writer:
        c = writer.add_connection("/front_laser", "sensor_msgs/msg/PointCloud2", typestore=store)
        for ns, row in rows:
            xyz = points(np.asarray(row[2:182], dtype=float))
            stamp = types["builtin_interfaces/msg/Time"](*divmod(ns, 10**9))
            header = types["std_msgs/msg/Header"](stamp, "front_laser")
            msg = types[c.msgtype](header, 1, len(xyz), fields, False, 12, xyz.nbytes, xyz.view(np.uint8).ravel(), True)
            writer.write(c, ns, store.serialize_cdr(msg, c.msgtype))


def prepare(source, output):
    raw = list(laser_rows(source/"intel.log.gz"))
    corrected = list(laser_rows(source/"intel.gfs.log.gz"))
    # Join the sparse external SLAM poses to their measured scans before choosing either window.
    elapsed = np.array([float(row[-1]) for row in raw])
    published = {}
    for row in corrected:
        index = int(np.argmin(abs(elapsed - float(row[-1]))))
        if (abs(elapsed[index] - float(row[-1])) < .001 and
                np.array_equal(np.asarray(row[2:182], float), np.asarray(raw[index][2:182], float))):
            if index in published:
                raise ValueError("Duplicate corrected pose for one raw scan")
            published[index] = np.asarray(row[-9:-6], float)
    anchors = []
    summary = {}
    for name, first in (("mapping", 169), ("inspection", 3420)):
        rows, skipped = [], 0
        indices = []
        for index, row in enumerate(raw[first:], first):
            if name == "mapping" and float(row[-1]) >= 220:
                break
            ns = int(Decimal(row[-3]) * 10**9)
            if rows and ns <= rows[-1][0]:
                skipped += 1
                continue
            rows.append((ns, row))
            indices.append(index)
            if name == "inspection" and len(rows) == 60:
                break
        anchors.append(published[first])
        with (output/f"{name}-external.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("t", "x", "y", "yaw"))
            for index in indices:
                if index in published:
                    writer.writerow([raw[index][-3], *relative(anchors[0], published[index])])
        write_window(output/name, rows)
        summary[name] = dict(scans=len(rows), skipped=skipped, start_ns=rows[0][0], end_ns=rows[-1][0])
    if summary["mapping"]["end_ns"] >= summary["inspection"]["start_ns"]:
        raise ValueError("Overlapping windows")
    profile = dict(lidar="/front_laser", lidar_frame="front_laser", odom=None, imu=None,
                   base_from_lidar_xyz_m=[0,0,0], base_from_lidar_xyzw=[0,0,0,1], height_band_m=[-.1,.1],
                   stationary_s=0, max_sync_gap_s=.5)
    (output/"profile.json").write_text(json.dumps(profile, indent=2))
    (output/"external-start.json").write_text(json.dumps(relative(*anchors).tolist()))
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        stage = Path(temp)/"data"
        stage.mkdir()
        prepare(args.source, stage)
        stage.rename(args.output)
