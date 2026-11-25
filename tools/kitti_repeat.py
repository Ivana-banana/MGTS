"""Evaluate the frozen KITTI Raw repeat-drive protocol for #161."""
import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.spatial import cKDTree

from mgts_nav.bag import Scan
from mgts_nav.coarse_start import estimate
from mgts_nav.grid import Grid, read_map
from mgts_nav.metrics import Pose, evaluate, wrap
from mgts_nav.slam import rotation, track

PREFIX = "2011_09_30/2011_09_30_drive_0027_sync/"


def timestamps(archive, sensor):
    lines = archive.read(PREFIX + sensor + "/timestamps.txt").decode().splitlines()
    return np.array(lines, dtype="datetime64[ns]").astype("int64") * 1e-9


def split_window(poses):
    tree = cKDTree(poses[:100, :2])
    candidates = []
    for first in range(len(poses)//2, len(poses)-39):
        xy = poses[first:first+40, :2]
        if np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() >= 5:
            candidates.append((tree.query(xy)[0].mean(), first))
    return min(candidates)[1]


def calibration(archive):
    lines = archive.read("2011_09_30/calib_imu_to_velo.txt").decode().splitlines()
    values = dict(line.split(":", 1) for line in lines if ":" in line)
    return np.fromstring(values["R"], sep=" ").reshape(3, 3), np.fromstring(values["T"], sep=" ")


def load_scan(archive, index, t, rot, offset):
    raw = archive.read(PREFIX + f"velodyne_points/data/{index:010d}.bin")
    xyz = np.frombuffer(raw, dtype="<f4").reshape(-1, 4)[:, :3]
    distance = np.linalg.norm(xyz[:, :2], axis=1)
    keep = (distance >= 2) & (distance <= 30) & (xyz[:, 2] >= -.8) & (xyz[:, 2] <= 1.2)
    ordered = xyz[keep][np.argsort(distance[keep], kind="stable")]
    angles = np.arctan2(ordered[:, 1], ordered[:, 0])
    _, ids = np.unique(np.floor(np.rad2deg(angles)*2), return_index=True)
    return Scan(t, ((ordered[ids]-offset) @ rot)[:, :2], (-offset @ rot)[:2], None, None)


def reference(packets):
    lat, lon = np.deg2rad(packets[:, :2]).T
    scale = 6378137 * np.cos(lat[0])
    xy = np.column_stack((lon, np.log(np.tan(np.pi/4 + lat/2)))) * scale
    yaw = packets[:, 5]
    return np.column_stack(((xy-xy[0]) @ rotation(yaw[0]), np.unwrap(yaw)-yaw[0]))


def integrated(times, packets, start):
    pose = start.copy()
    poses = [pose.copy()]
    for i, dt in enumerate(np.diff(times)):
        # KITTI OXTS vf, vl and wu in the vehicle frame.
        vf, vl, wu = packets[i:i+2, [8, 9, 22]].mean(axis=0)
        pose[:2] += rotation(pose[2] + wu*dt/2) @ np.array([vf, vl]) * dt
        pose[2] += wu*dt
        poses.append(pose.copy())
    return np.array(poses)


def report(times, truth, trajectory, statuses):
    reference_rows = [Pose(t, *p) for t, p in zip(times, truth, strict=True)]
    estimates = [Pose(t, *p) for t, p in zip(times, trajectory, strict=True)]
    result = evaluate(estimates, reference_rows)
    delta = trajectory[-1] - truth[-1]
    result.update(final_xy_m=float(np.linalg.norm(delta[:2])), final_yaw_deg=float(np.rad2deg(abs(wrap(delta[2])))),
                  prediction_only_fraction=statuses.count("prediction_only")/len(times))
    return result


def write_poses(path, times, poses):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(Pose._fields)
        writer.writerows((t, *p) for t, p in zip(times, poses, strict=True))


def run(directory, output):
    output.mkdir(parents=True, exist_ok=False)
    with ZipFile(directory / "drive.zip") as archive, ZipFile(directory / "calib.zip") as calib:
        lidar_times = timestamps(archive, "velodyne_points")
        imu_times = timestamps(archive, "oxts")
        packets = np.array([np.fromstring(archive.read(PREFIX + f"oxts/data/{i:010d}.txt").decode(),
                                          sep=" ") for i in range(len(imu_times))])
        poses = reference(packets)
        first = split_window(poses)
        rot, offset = calibration(calib)
        windows = [range(100), range(first, first+40)]
        mapping, inspection = [[load_scan(archive, i, lidar_times[i], rot, offset) for i in window]
                               for window in windows]
    # Synced raw packets are paired by frame index. Report the timestamp mismatch explicitly.
    result = {"inspection_first": first}
    grid = Grid(.12)
    mapped = list(track(mapping, np.zeros(3), grid=grid))
    grid.save(output)
    pixels, meta, points = read_map(output / "map.yaml")
    for name, window in zip(("mapping", "inspection"), windows):
        times = lidar_times[window]
        truth = poses[window]
        data = packets[window]
        result[name] = dict(distance_m=float(np.linalg.norm(np.diff(truth[:, :2], axis=0), axis=1).sum()),
                            duration_s=float(times[-1]-times[0]),
                            max_sync_skew_s=float(np.max(abs(times-imu_times[window]))),
                            turning_frames=int(np.sum(abs(data[:, 22]) >= .1)),
                            moving_frames=int(np.sum(np.hypot(data[:, 8], data[:, 9]) >= 1)))
        write_poses(output / f"{name}-reference.csv", times, truth)
        if name == "mapping":
            result[name]["score"] = report(times, truth, [r.pose for r in mapped], [r.status for r in mapped])
    starts = {"known": truth[0]}
    try:
        starts["blind"] = estimate(inspection[0], pixels, meta, {})
    except ValueError as exc:
        result["blind"] = dict(refused=True, reason=str(exc), coverage=0, start_refusal_fraction=1)
    for name, start in starts.items():
        rows = list(track(inspection, start, map_points=points))
        result[name] = report(times, truth, [r.pose for r in rows], [r.status for r in rows])
        result[name]["start_refusal_fraction"] = 0
        write_poses(output / f"{name}.csv", times, [r.pose for r in rows])
    result["kinematics"] = report(times, truth, integrated(imu_times[window], data, truth[0]), ["integrated"]*len(times))
    (output / "scores.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.archives, args.output), indent=2))
