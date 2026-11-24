"""Compare gravity leveling on paired ROS 2 recordings of a tilting robot."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

from mgts_nav.bag import load_config, reader, scans, stamp
from mgts_nav.demo import DURATION, PROFILE, START_NS, WALLS, cross, generate, pose_at
from mgts_nav.grid import read_map
from mgts_nav.leveling import GRAVITY
from mgts_nav.map_eval import evaluate_map, segment_distances
from mgts_nav.metrics import evaluate, read_csv, wrap
from mgts_nav.slam import world

FLOOR, CEILING = -.45, 2.4  # metres, relative to the constant height of the body origin
LIDAR_MOUNT = Rotation.from_euler("ZYX", [19., -3., 4.], degrees=True)
IMU_MOUNT = Rotation.from_euler("ZYX", [-22., 14., 9.], degrees=True)
BEARING, ELEVATION = np.meshgrid(np.linspace(-np.pi, np.pi, 160, endpoint=False),
                                  np.deg2rad([-12., 0., 12.]))
RAYS = np.column_stack(((np.cos(ELEVATION)*np.cos(BEARING)).ravel(),
                        (np.cos(ELEVATION)*np.sin(BEARING)).ravel(), np.sin(ELEVATION).ravel()))


def attitude(run, t, amplitude):
    duration = DURATION[run]
    # Fade in and out at the boundaries of the two stationary parts of the demo.
    ramp = np.clip(min(t-1., duration-1.-t)/.4, 0., 1.)
    angle = np.deg2rad(amplitude) * (.5-.5*np.cos(np.pi*ramp))
    roll = angle * np.sin(2*np.pi*.8*(t-1.))
    pitch = .7 * angle * np.sin(2*np.pi*.6*(t-1.) + .45)
    return Rotation.from_euler("ZYX", [pose_at(run, t)[2], pitch, roll])


def imu_ideal(run, t, amplitude):
    half = .0001
    prev, nxt = (attitude(run, t+d, amplitude) for d in (-half, half))
    omega = (prev.inv()*nxt).as_rotvec()/(2*half)
    dt = .05
    p = pose_at(run, t)[:2]
    acc = (pose_at(run, t+dt)[:2] - 2*p + pose_at(run, t-dt)[:2])/dt**2
    specific = attitude(run, t, amplitude).inv().apply([*acc, GRAVITY])
    return IMU_MOUNT.inv().apply(omega), IMU_MOUNT.inv().apply(specific)


def render(run, t, amplitude, cfg, rng):
    rot = attitude(run, t, amplitude)
    origin = np.r_[pose_at(run, t)[:2], 0.] + rot.apply(cfg["base_from_lidar_xyz_m"])
    directions = (rot*LIDAR_MOUNT).apply(RAYS)
    delta = WALLS[:, 0] - origin[:2]
    edges = WALLS[:, 1] - WALLS[:, 0]
    det = cross(directions[:, None, :2], edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = cross(delta, edges)/det
        along = cross(delta, directions[:, None, :2])/det
        plane = (np.where(directions[:, 2] > 0, CEILING, FLOOR) - origin[2])/directions[:, 2]
    # The first floor/ceiling intersection clips the vertical walls.
    wall_hits = (abs(det) > 1e-10) & (distance > 0) & (along >= 0) & (along <= 1)
    first = np.minimum(np.where(wall_hits, distance, np.inf).min(axis=1),
                       np.where(plane > 0, plane, np.inf))
    noise = rng.normal(0., .012, len(RAYS))
    keep = (rng.random(len(RAYS)) >= .1) & np.isfinite(first)
    return ((first[keep]+noise[keep])[:, None] * RAYS[keep]).astype("<f4")


def make_bag(source, target, run, amplitude, cfg, seed, *, gap=False, end=None):
    rng = np.random.default_rng(seed)
    store = get_typestore(Stores.ROS2_HUMBLE)
    with reader(source) as bag, Writer(target, version=8) as writer:
        conns = {c.id: writer.add_connection(c.topic, c.msgtype, typestore=store)
                 for c in bag.connections}
        for c, received, raw in bag.messages():
            msg = bag.deserialize(raw, c.msgtype)
            t = stamp(msg)-START_NS[run]/1e9
            if end is not None and t > end:
                continue
            if c.topic == cfg["lidar"]:
                xyz = render(run, t, amplitude, cfg, rng)
                msg.width, msg.row_step, msg.data = len(xyz), xyz.nbytes, xyz.view(np.uint8).ravel()
                raw = store.serialize_cdr(msg, c.msgtype)
            elif c.topic == cfg["imu"]:
                rate, force = imu_ideal(run, t, amplitude)
                rate += np.array([.008, -.005, .012]) + rng.normal(0., .001, 3)
                force += rng.normal(0., .015, 3)
                if gap and 3. <= t < 4.:
                    continue
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = rate
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = force
                raw = store.serialize_cdr(msg, c.msgtype)
            writer.write(conns[c.id], received, raw)


def nav(*args):
    subprocess.run([sys.executable, "-m", "mgts_nav", *map(str, args)], check=True)


def measure(result, bag, profile, reference):
    rows = read_csv(result/"trajectory.csv")
    truth = reference
    np.testing.assert_allclose([r.t for r in rows], [r.t for r in truth], atol=1e-8, rtol=0)
    trajectory = evaluate(rows, truth)
    status = json.loads((result/"summary.json").read_text())["status"]
    residuals, yaw_steps, unavailable = [], [], []
    previous_yaw = None
    for scan, ref in zip(scans(bag, load_config(profile)), truth, strict=True):
        if len(scan.xy):
            residuals.extend(segment_distances(world(scan.xy, np.asarray(ref[1:])), WALLS))
        if scan.empty_reason == "leveling_unavailable":
            unavailable.append(scan.t)
        if previous_yaw is not None and scan.gyro_delta is not None:
            yaw_steps.append(wrap(scan.gyro_delta-wrap(ref.yaw-previous_yaw)))
        previous_yaw = ref.yaw
    return dict(scans=len(rows), prediction_only=status.get("prediction_only", 0),
                prediction_fraction=status.get("prediction_only", 0)/len(rows),
                ate_m=trajectory["ate_xy_rmse_m"], scan_wall_rmse_m=float(np.sqrt(np.mean(np.square(residuals)))),
                gyro_step_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(np.square(yaw_steps))))),
                gyro_intervals=len(yaw_steps), leveling_unavailable=len(unavailable),
                unavailable_bounds=([unavailable[0], unavailable[-1]] if unavailable else []))


def experiment(root):
    source = root/"source"
    seed = 41
    generate(source, seed=seed)
    cfg = dict(load_config(PROFILE), base_from_lidar_xyzw=LIDAR_MOUNT.as_quat().tolist(),
               base_from_imu_xyzw=IMU_MOUNT.as_quat().tolist())
    truth = {run: read_csv(source/f"{run}_truth.csv") for run in ("mapping", "inspection")}
    profiles = {}
    for mode in ("off", "on"):
        profiles[mode] = root/f"{mode}.json"
        profiles[mode].write_text(json.dumps(dict(cfg, gravity_level=mode == "on")), encoding="utf-8")
    initial = None
    report = {}
    for amplitude in (0, 7, 15):
        case = root/f"roll-{amplitude}"
        case.mkdir()
        for i, run in enumerate(("mapping", "inspection")):
            make_bag(source/run, case/run, run, amplitude, cfg, seed+i)
        if amplitude == 15:
            for name, end in (("gap-rest", None), ("gap-moving", 8.5)):
                make_bag(source/"inspection", case/name, "inspection", amplitude, cfg, seed+1, gap=True, end=end)
        for mode, profile in profiles.items():
            dest = case/mode
            dest.mkdir()
            nav("map", case/"mapping", "--config", profile, "--out", dest/"map")
            info = measure(dest/"map", case/"mapping", profile, truth["mapping"])
            _, meta, occupied = read_map(dest/"map/map.yaml")
            info["map_wall_rmse_m"] = float(np.sqrt(np.mean(segment_distances(occupied, WALLS)**2)))
            info["wall_coverage"] = evaluate_map(occupied, WALLS, meta["resolution"], .12)["static_wall_length_covered_fraction"]
            record = {"mapping": info}
            for name in (("inspection", "gap-rest", "gap-moving") if amplitude == 15 else ("inspection",)):
                start_flags = [] if initial is None else ["--initial", *initial]
                nav("localize", case/name, "--config", profile, "--map", dest/"map/map.yaml",
                    "--out", dest/name, *start_flags)
                start = json.loads((dest/name/"summary.json").read_text())["initial_pose"]
                if initial is None:
                    initial = start
                np.testing.assert_allclose(start, initial, atol=1e-12, rtol=0)
                reference = truth["inspection"]
                if name == "gap-moving":
                    reference = [p for p in reference if p.t <= START_NS["inspection"]/1e9 + 8.5]
                record[name] = measure(dest/name, case/name, profile, reference)
            report[f"{amplitude}/{mode}"] = record
    report["shared_initial_pose"] = initial
    (root/"metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    target = parser.parse_args().out
    if target.exists():
        parser.error("Choose a new output directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as tmp:
        stage = Path(tmp)/"results"
        stage.mkdir()
        experiment(stage)
        stage.rename(target)
    print(target/"metrics.json")
