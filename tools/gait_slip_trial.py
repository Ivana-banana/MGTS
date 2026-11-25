"""Measure gait, slip and tilt on the office wall model and a real OXTS attitude control."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from mgts_nav.bag import Scan
from mgts_nav.demo import WALLS, lidar_points
from mgts_nav.grid import Grid, read_map
from mgts_nav.leveling import GRAVITY, build_leveler
from mgts_nav.map_eval import segment_distances
from mgts_nav.slam import match, relative, rotation, track, world
from tools.kitti_repeat import PREFIX, calibration, load_scan, reference, report, timestamps


def trajectory(t):
    x = .55 * np.clip(t-1, 0, 2)
    y = .4 * np.clip(t-3, 0, 2)
    yaw = .75 * np.clip(t-5, 0, 2)
    return np.array([x, y, yaw])


def attitude(t):
    active = np.clip(min((t-1)*4, (7-t)*4), 0, 1)
    roll = active*np.deg2rad(12)*np.sin(2*np.pi*1.6*t)
    pitch = active*np.deg2rad(8)*np.cos(2*np.pi*1.6*t)
    return Rotation.from_euler("ZYX", [trajectory(t)[2], pitch, roll])


def synthetic():
    times = np.arange(901)/100
    gyro = np.array([(attitude(t-.0001).inv()*attitude(t+.0001)).as_rotvec()/.0002 for t in times])
    acc = []
    for t in times:
        a = (trajectory(t+.05)-2*trajectory(t)+trajectory(t-.05))[:2]/.05**2
        step = 2*np.sin(2*np.pi*1.6*t) if 1 < t < 7 else 0
        acc.append(attitude(t).inv().apply([*a, GRAVITY+step]))
    leveler = build_leveler(times, gyro, acc, .5, .15)
    profile = {"base_from_lidar_xyz_m": [0, 0, 0], "base_from_lidar_xyzw": [0, 0, 0, 1]}
    rng = np.random.default_rng(169)
    times = times[::10]
    truth = np.array([trajectory(t) for t in times])
    raw, leveled, slips = [], [], []
    for i, (t, p) in enumerate(zip(times, truth)):
        xyz = lidar_points(p, t, profile, rng)
        tilt = Rotation.from_euler("z", -p[2])*attitude(t)
        xyz = tilt.inv().apply(xyz)
        odo = p.copy()
        # A slipping encoder accumulates 0.6 m lateral error over three increments.
        odo[1] += .2*np.clip(i-38, 0, 3)
        slips.append(38 < i <= 41)
        for series, cloud in ((raw, xyz), (leveled, leveler.at(t).apply(xyz))):
            points = cloud[(cloud[:, 2] >= .15) & (cloud[:, 2] <= 1.5), :2]
            series.append(Scan(t, points, np.zeros(2), odo.copy(), None))
    return times, truth, raw, leveled, np.array(slips)


def gated(scans):
    corrected = scans[0].odom.copy()
    result = [replace(scans[0], odom=corrected.copy())]
    flags = [False]
    for prev, current in zip(scans, scans[1:]):
        step = relative(prev.odom, current.odom)
        lidar, support, ok = match(current.xy, cKDTree(prev.xy), np.zeros(3))
        reject = ok and support >= .65 and np.linalg.norm(step[:2]-lidar[:2]) > .12
        if reject:
            step[:2] = 0.
        corrected = np.r_[world(step[:2], corrected), corrected[2]+step[2]]
        result.append(replace(current, odom=corrected.copy()))
        flags.append(bool(reject))
    return result, np.array(flags)


def office(dest):
    times, truth, raw, leveled, slips = synthetic()
    corrected, flags = gated(leveled)
    result = {}
    for mode, stream in (("default", raw), ("level", leveled), ("gated", corrected)):
        grid = Grid(.08)
        rows = list(track(stream, truth[0], grid=grid))
        directory = dest/mode
        directory.mkdir()
        grid.save(directory)
        points = read_map(directory/"map.yaml")[2]
        scores = {"map_wall_p90_m": float(np.quantile(segment_distances(points, WALLS), .9))}
        for name, low, high in (("straight", 1, 3), ("lateral", 3, 5), ("turn", 5, 7), ("stop", 7, 9.1)):
            idx = np.flatnonzero((times >= low) & (times < high))
            poses = [rows[i].pose for i in idx]
            wall = np.concatenate([segment_distances(world(stream[i].xy, rows[i].pose), WALLS) for i in idx])
            applied = flags[idx] if mode == "gated" else np.zeros(len(idx), dtype=bool)
            scores[name] = dict(report(times[idx], truth[idx], poses, [rows[i].status for i in idx]),
                                wall_p90_m=float(np.quantile(wall, .9)),
                                false_disabled=int(np.sum(applied & ~slips[idx])),
                                clean_steps=int(np.sum(~slips[idx])), disabled=int(applied.sum()))
        result[mode] = scores
    return result


def field(archives):
    # Isolate the direction of the projection effect using recorded independent OXTS tilt.
    with ZipFile(archives/"drive.zip") as drive, ZipFile(archives/"calib.zip") as calib:
        transform = calibration(calib)
        times = timestamps(drive, "velodyne_points")[:100]
        imu = np.array([np.fromstring(drive.read(PREFIX+f"oxts/data/{i:010d}.txt").decode(), sep=" ")
                        for i in range(len(times))])
        truth = reference(imu)
        basic, level = [], []
        for i, t in enumerate(times):
            # Apply full extrinsics to the same selected 3D returns before leveling.
            xyz = np.frombuffer(drive.read(PREFIX+f"velodyne_points/data/{i:010d}.bin"), "<f4").reshape(-1,4)[:, :3]
            d = np.linalg.norm(xyz[:, :2], axis=1)
            mask = (d >= 2) & (d <= 30) & (xyz[:, 2] >= -.8) & (xyz[:, 2] <= 1.2)
            ordered = xyz[mask][np.argsort(d[mask], kind="stable")]
            _, ids = np.unique(np.floor(np.rad2deg(np.arctan2(ordered[:, 1], ordered[:, 0]))*2), return_index=True)
            body = (ordered[ids]-transform[1]) @ transform[0]
            sc = load_scan(drive, i, t, *transform)
            basic.append(sc)
            tilt = Rotation.from_euler("xyz", [*imu[i, 3:5], 0])
            level.append(replace(sc, xy=tilt.apply(body)[:, :2],
                                 origin=tilt.apply(-transform[1] @ transform[0])[:2]))
    scores = {}
    for mode, stream in (("raw", basic), ("attitude_control", level)):
        rows = list(track(stream, np.zeros(3), grid=Grid(.12)))
        scores[mode] = report(times, truth, [r.pose for r in rows], [r.status for r in rows])
    scores["max_abs_roll_pitch_deg"] = np.rad2deg(np.abs(imu[:, 3:5]).max(axis=0)).tolist()
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--archives", type=Path, default=Path("out/kitti161"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scores = {"office": office(args.output), "kitti": field(args.archives)}
    (args.output/"scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
