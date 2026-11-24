"""Prepare and independently reference two nonadjacent visits to HEAP-1."""
import argparse
import csv
import io
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import numpy as np
from numcodecs import blosc
from rosbags.typesys import Stores, get_typestore
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
import yaml

from heap1_sample import ROOT, read_window, write_bag


VISITS = {"mapping": (150, 190), "inspection": (2560, 2620)}


def stream_arrays(name, keys):
    # These auxiliary streams contain a single Blosc chunk per array.
    with urlopen(f"{ROOT}/data/{name}.tar", timeout=90) as response:
        payload = response.read(32 * 1024**2 + 1)
    if len(payload) > 32 * 1024**2:
        raise ValueError(f"Oversized auxiliary archive: {name}")
    arrays = {}
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        attrs = json.load(archive.extractfile(f"{name}/.zattrs"))
        for key in keys:
            prefix = f"{name}/{key}/"
            m = json.load(archive.extractfile(prefix + ".zarray"))
            if any(n > c for n, c in zip(m["shape"], m["chunks"], strict=True)):
                raise ValueError("Expected a single chunk")
            member = prefix + ".".join("0" for _ in m["shape"])
            raw = blosc.decompress(archive.extractfile(member).read())
            a = np.frombuffer(raw, m["dtype"]).reshape(m["chunks"])
            arrays[key] = a[:m["shape"][0]].copy()
            del raw, a
    if not (np.diff(arrays["timestamp"]) > 0).all():
        raise ValueError(f"Unordered sensor time: {name}")
    return attrs, arrays


def extrinsics(transforms, parent, child):
    tf = transforms[child]
    if (tf["base_frame_id"], tf["child_frame_id"]) != (parent, child):
        raise ValueError(f"Missing transform {parent} <- {child}")
    return ([tf["translation"][k] for k in "xyz"], [tf["rotation"][k] for k in "xyzw"])


def reference(times, transforms):
    # Validate which rigid body the ENU pose describes using the inverse published TF.
    attrs, nav = stream_arrays("cpt7_ie_tc_odometry", ["timestamp", "pose_pos", "pose_orien"])
    tf_attrs, tf = stream_arrays("cpt7_ie_tc_tf", ["timestamp", "translation", "rotation"])
    if attrs["frame_id"] != "enu_origin" or tf_attrs["frame_id"] != "cpt7_imu":
        raise ValueError("Unexpected CPT7 frames")
    q = Rotation.from_quat(nav["pose_orien"])
    np.testing.assert_array_equal(nav["timestamp"], tf["timestamp"])
    np.testing.assert_allclose(nav["pose_pos"] + q.apply(tf["translation"]), 0, atol=1e-8)
    np.testing.assert_allclose((q * Rotation.from_quat(tf["rotation"])).magnitude(), 0, atol=1e-8)
    clock = nav["timestamp"]
    if times[0] < clock[0] or times[-1] > clock[-1] or np.diff(clock).max() > .05:
        raise ValueError("Insufficient independent pose coverage")
    enu_sensor = Slerp(clock, q)(times)
    sensor_xyz = np.column_stack([np.interp(times, clock, c) for c in nav["pose_pos"].T])
    base_box, base_box_q = extrinsics(transforms, "base", "box_base")
    box_sensor, box_sensor_q = extrinsics(transforms, "box_base", "cpt7_imu")
    base_sensor = base_box + Rotation.from_quat(base_box_q).apply(box_sensor)
    base_sensor_q = Rotation.from_quat(base_box_q) * Rotation.from_quat(box_sensor_q)
    enu_base = enu_sensor * base_sensor_q.inv()
    pos = sensor_xyz - enu_base.apply(base_sensor)
    yaw = enu_base.as_euler("ZYX")[:, 0]
    # The planar map origin has the position and heading of the first mapping scan.
    aligned = Rotation.from_euler("z", -yaw[0]).apply(pos - pos[0])
    return np.column_stack((aligned[:, :2], np.arctan2(np.sin(yaw-yaw[0]), np.cos(yaw-yaw[0]))))


def prepare(stage):
    with urlopen(f"{ROOT}/metadata/tf.yaml", timeout=30) as reply:
        transforms = yaml.safe_load(reply)
    xyz, quat = extrinsics(transforms, "base", "velodyne_lidar")
    _, imu_quat = extrinsics(transforms, "base", "anymal_imu")
    imu_attrs, imu = stream_arrays("anymal_imu", ["timestamp", "ang_vel", "lin_acc"])
    if imu_attrs["frame_id"] != "anymal_imu" or imu_attrs["transform"] != transforms["anymal_imu"]:
        raise ValueError("Inconsistent IMU calibration")
    profile = dict(lidar="/velodyne_points", lidar_frame="velodyne_lidar", base_frame="base",
                   base_from_lidar_xyz_m=xyz, base_from_lidar_xyzw=quat, imu="/anymal_imu",
                   imu_frame="anymal_imu", base_from_imu_xyzw=imu_quat, odom=None,
                   stationary_s=0, max_sync_gap_s=.05, height_band_m=[.15, 1.5])
    (stage / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    first, end = VISITS["mapping"][0], VISITS["inspection"][1]
    times = read_window("timestamp", "<f8", window=slice(first, end))
    valid = read_window("valid", "<u4", window=slice(first, end))
    truth = reference(times, transforms)
    clock = imu["timestamp"]
    store = get_typestore(Stores.ROS2_HUMBLE)
    details = {}
    for bag, (start, stop) in VISITS.items():
        window = slice(start-first, stop-first)
        t = times[window]
        chunk, offset = divmod(start, 512)
        points = read_window("points" if chunk == 0 else f"points{chunk}", "<f4", (35000, 3),
                             slice(offset, offset+len(t)))
        lo, hi = clock.searchsorted(t[[0, -1]])
        if lo == 0 or hi == len(clock):
            raise ValueError("IMU does not bracket the window")
        cut = slice(lo-1, hi+1)
        gaps = np.diff(clock[cut])
        if gaps.max() > profile["max_sync_gap_s"]:
            raise ValueError("IMU synchronization gap")
        samples = zip(clock[cut], imu["ang_vel"][cut], imu["lin_acc"][cut], strict=True)
        write_bag(stage/bag, store, profile["lidar"], profile["lidar_frame"], t, points,
                  valid[window], (profile["imu_frame"], samples))
        with (stage/f"{bag}-reference.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["t", "x", "y", "yaw"])
            writer.writerows(np.column_stack((t, truth[window])))
        details[bag] = dict(start=t[0], end=t[-1], imu_max_gap_s=gaps.max())
    ref_map = truth[:VISITS["mapping"][1]-first, :2]
    ref_inspection = truth[VISITS["inspection"][0]-first:, :2]
    details["max_distance_from_mapping_start_m"] = np.linalg.norm(truth[:, :2], axis=1).max()
    details["max_inspection_distance_to_mapping_track_m"] = cKDTree(ref_map).query(ref_inspection)[0].max()
    (stage/"reference-summary.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(json.dumps(details))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    output = parser.parse_args().out
    if output.exists():
        parser.error("Choose a new output directory")
    output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp)/"dataset"
        stage.mkdir()
        prepare(stage)
        stage.rename(output)
