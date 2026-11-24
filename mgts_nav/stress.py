"""Replay controlled sensor faults on copies of the synthetic inspection bag."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from .bag import load_config, reader, scans, stamp
from .demo import PROFILE
from .grid import Grid, read_map
from .metrics import Pose, evaluate, read_csv
from .slam import track


CASES = ("clean", "lidar_no_returns", "lidar_no_messages", "imu_gap", "odom_drift")


def fault_bag(source, target, case, cfg):
    """Only intended for the three standard message types in the demo."""
    store = get_typestore(Stores.ROS2_HUMBLE)
    with reader(source) as bag, Writer(target, version=8) as writer:
        outputs = {c.id: writer.add_connection(c.topic, c.msgtype, typestore=store)
                   for c in bag.connections}
        start = None
        affected = 0
        for conn, received, raw in bag.messages():
            msg = bag.deserialize(raw, conn.msgtype)
            if start is None:
                start = stamp(msg)
            elapsed = stamp(msg) - start
            topic = conn.topic
            if 3 <= elapsed < 5:
                if ((case == "lidar_no_messages" and topic == cfg["lidar"])
                        or (case == "imu_gap" and topic == cfg["imu"])):
                    affected += 1
                    continue
                if case == "lidar_no_returns" and topic == cfg["lidar"]:
                    msg.width = msg.row_step = 0
                    msg.data = np.empty(0, dtype=np.uint8)
                    raw = store.serialize_cdr(msg, conn.msgtype)
                    affected += 1
            if case == "odom_drift" and topic == cfg["odom"] and elapsed >= 3:
                # Add a constant velocity error along odom-frame x.
                msg.pose.pose.position.x += 0.30 * (elapsed - 3)
                raw = store.serialize_cdr(msg, conn.msgtype)
                affected += 1
            writer.write(outputs[conn.id], received, raw)
    return affected


def summarize(estimates, truth):
    if not estimates:
        raise ValueError("No inspection poses")
    poses = [Pose(row.t, *row.pose) for row in estimates]
    metrics = evaluate(poses, truth)
    run = longest = matched = 0
    for row in estimates:
        matched += row.status == "matched"
        run = run + 1 if row.status == "prediction_only" else 0
        longest = max(longest, run)
    return dict(ate_xy_rmse_m=metrics["ate_xy_rmse_m"], truth_coverage=metrics["coverage"],
                estimates=len(poses), matched_fraction=matched / len(poses),
                max_prediction_only_scans=longest,
                max_pose_interval_s=max((b.t - a.t for a, b in zip(poses, poses[1:])), default=None))


def run(demo, config=PROFILE, map_path=None):
    cfg = load_config(config)
    initial = json.loads((demo / "initial_pose.json").read_text(encoding="utf-8"))
    with TemporaryDirectory() as work:
        work = Path(work)
        if map_path is None:
            grid = Grid(0.08)
            for _ in track(scans(demo / "mapping", cfg), np.zeros(3), grid=grid):
                pass
            grid.save(work)
            map_path = work / "map.yaml"
        points = read_map(map_path)[2]
        results = {}
        for case in CASES:
            source = demo / "inspection"
            affected = 0
            if case != "clean":
                altered = work / case
                affected = fault_bag(source, altered, case, cfg)
                source = altered
            estimates = list(track(scans(source, cfg), initial, map_points=points))
            results[case] = summarize(estimates, read_csv(demo / "inspection_truth.csv"))
            results[case]["affected_messages"] = affected
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", type=Path)
    parser.add_argument("--config", type=Path, default=PROFILE)
    parser.add_argument("--map", type=Path, help="Reuse a map built from the same mapping bag")
    args = parser.parse_args()
    print(json.dumps(run(args.demo, args.config, args.map), indent=2, allow_nan=False))
