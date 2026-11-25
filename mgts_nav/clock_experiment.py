"""Measure signed IMU header offsets on copies of the default office demo."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from .bag import load_config, reader, scans
from .demo import PROFILE
from .grid import Grid, read_map
from .metrics import read_csv
from .slam import track


def shift_imu(source, output, topic, offset_ns):
    store = get_typestore(Stores.ROS2_HUMBLE)
    with reader(source) as bag, Writer(output, version=8) as writer:
        conns = {c.id: writer.add_connection(c.topic, c.msgtype, typestore=store)
                 for c in bag.connections}
        for c, received, raw in bag.messages():
            if c.topic == topic:
                m = bag.deserialize(raw, c.msgtype)
                stamp = m.header.stamp
                ns = stamp.sec * 10**9 + stamp.nanosec + offset_ns
                stamp.sec, stamp.nanosec = divmod(ns, 10**9)
                raw = store.serialize_cdr(m, c.msgtype)
            writer.write(conns[c.id], received, raw)


def measure(bag, cfg, start, points, truth):
    stream = list(scans(bag, cfg))
    rows = list(track(stream, start, map_points=points))
    np.testing.assert_allclose([r.t for r in rows], truth[:, 0], rtol=0, atol=1e-8)
    poses = np.stack([r.pose for r in rows])
    angle_error = np.rad2deg(np.angle(np.exp(1j * (poses[:, 2] - truth[:, 3]))))
    xy_error = np.linalg.norm(poses[:, :2] - truth[:, 1:3], axis=1)
    yaw_delta = np.diff(truth[:, 3], prepend=np.nan)
    gyro = np.array([s.gyro_delta if s.gyro_delta is not None else np.nan for s in stream])
    matched = np.array([r.status == "matched" for r in rows])
    results = {}
    for label, mask in (("turning", np.abs(yaw_delta) > 1e-9), ("stationary", np.abs(yaw_delta) <= 1e-9)):
        usable = mask & np.isfinite(gyro)
        results[label] = dict(scans=int(sum(mask)), matched=float(matched[mask].mean()),
                              ate_m=float(np.sqrt(np.mean(xy_error[mask]**2))),
                              yaw_rmse_deg=float(np.sqrt(np.mean(angle_error[mask]**2))),
                              gyro_intervals=int(sum(usable)),
                              gyro_step_rmse_deg=float(np.rad2deg(np.sqrt(np.mean((gyro[usable]-yaw_delta[usable])**2)))))
    return poses, matched, results


def run(demo):
    cfg = load_config(PROFILE)
    truth = np.array(read_csv(demo/"inspection_truth.csv"))
    start = json.loads((demo/"initial_pose.json").read_text(encoding="utf-8"))
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        grid = Grid(.08)
        list(track(scans(demo/"mapping", cfg), np.zeros(3), grid=grid))
        grid.save(work)
        points = read_map(work/"map.yaml")[2]
        baseline, status, scores = measure(demo/"inspection", cfg, start, points, truth)
        results = {"original": scores}
        for shift in (-.25, .25):
            target = work/f"shift{shift}"
            shift_imu(demo/"inspection", target, cfg["imu"], round(shift*10**9))
            for correction in (0., -shift):
                corrected = dict(cfg, header_offset_s={"imu": correction})
                poses, matched, scores = measure(target, corrected, start, points, truth)
                if correction:
                    np.testing.assert_allclose(poses, baseline, atol=1e-8, rtol=0)
                    np.testing.assert_array_equal(matched, status)
                results[f"shift={shift}, correction={correction}"] = scores
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", type=Path, help="Default office demo with synthetic truth")
    print(json.dumps(run(parser.parse_args().demo), indent=2, allow_nan=False))
