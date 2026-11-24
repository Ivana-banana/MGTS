"""Convert non-overlapping HEAP-1 Velodyne windows to lidar-only ROS 2 bags."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

import numpy as np
import yaml
from numcodecs import blosc
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

REVISION = "af6b6dd6f91696f8ec8c98134e195514834911b2"
ROOT = ("https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset/resolve/"
        f"{REVISION}/2024-11-14-13-45-37")
# TAR payload offsets, lengths and SHA-256 at this immutable revision.
CHUNKS = {
    "points": (75065344, 88887497, "dac6e722351bfbb6f12937d3d8c9a637015755b7b579add742b9e49dee6e89ff"),
    "timestamp": (682354688, 1089489, "6f83143bfc293cb947e9337e2144a07820dae8481102b22bb8279a2b84950e57"),
    "valid": (683449344, 1088907, "43aadcdac0d215a9f5f719d90b99a3afa2c9f987d906e05cca7a0fe5b1af9306"),
}
WINDOW = slice(300, 360)
SPLIT = 40


def read_window(name, dtype, row_shape=()):
    offset, size, digest = CHUNKS[name]
    end = offset + size - 1
    request = Request(f"{ROOT}/data/velodyne_points.tar?chunk={name}",
                      headers={"Range": f"bytes={offset}-{end}"})
    with urlopen(request, timeout=90) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {offset}-{end}/684544000":
            raise ValueError(f"Server did not return the requested range for {name}")
        payload = response.read(size + 1)
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"Unexpected {name} chunk at pinned dataset revision")
    array = np.frombuffer(blosc.decompress(payload), dtype).reshape((-1, *row_shape))
    return array[WINDOW].copy()


def write_bag(path, store, topic, frame, timestamps, points, valid):
    types = store.types
    field = types["sensor_msgs/msg/PointField"]
    fields = [field(name, 4 * idx, 7, 1) for idx, name in enumerate("xyz")]
    with Writer(path, version=8) as bag:
        conn = bag.add_connection(topic, "sensor_msgs/msg/PointCloud2", typestore=store)
        for t, xyz, count in zip(timestamps, points, valid, strict=True):
            if not 0 < count <= len(xyz):
                raise ValueError(f"Invalid point count at {t}")
            xyz = xyz[:count]
            ns = round(float(t) * 1e9)
            stamp = types["builtin_interfaces/msg/Time"](*divmod(ns, 1_000_000_000))
            header = types["std_msgs/msg/Header"](stamp, frame)
            cloud = types[conn.msgtype](header, 1, len(xyz), fields, False, 12, 12 * len(xyz),
                                         xyz.view(np.uint8).ravel(), bool(np.isfinite(xyz).all()))
            bag.write(conn, ns, store.serialize_cdr(cloud, conn.msgtype))


def convert(output):
    if output.exists():
        raise FileExistsError(f"Choose a new directory: {output}")
    timestamps = read_window("timestamp", "<f8")
    if not np.isfinite(timestamps).all() or not (np.diff(timestamps) > 0).all():
        raise ValueError("Invalid Velodyne timestamps")
    valid = read_window("valid", "<u4")
    points = read_window("points", "<f4", (35000, 3))
    with urlopen(f"{ROOT}/metadata/velodyne_points.yaml", timeout=30) as response:
        meta = yaml.safe_load(response)
    transform = meta["transform"]
    frame = meta["frame_id"]
    if transform["child_frame_id"] != frame:
        raise ValueError("Unexpected calibration frame")
    topic = "/" + meta["topic"].lstrip("/")
    profile = dict(lidar=topic, lidar_frame=frame, base_frame=transform["base_frame_id"],
                   base_from_lidar_xyz_m=[transform["translation"][k] for k in "xyz"],
                   base_from_lidar_xyzw=[transform["rotation"][k] for k in "xyzw"],
                   imu=None, odom=None, height_band_m=[0.15, 1.5], stationary_s=0, max_sync_gap_s=0.35)
    store = get_typestore(Stores.ROS2_HUMBLE)
    output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "dataset"
        stage.mkdir()
        (stage / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
        for bag, window in (("mapping", slice(None, SPLIT)), ("inspection", slice(SPLIT, None))):
            write_bag(stage / bag, store, topic, frame, timestamps[window], points[window], valid[window])
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    convert(parser.parse_args().out)
