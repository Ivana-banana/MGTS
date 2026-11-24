"""Audit solve products against the input lidar clocks and exported ROS messages."""
import argparse
from collections import Counter
import csv
from dataclasses import astuple
import json
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReaderError
from rosbags.typesys import Stores, get_typestore

from .bag import check_frame, connections, load_config, reader, stamp
from .grid import read_map
from .metrics import read_csv
from .ros_export import _read_poses


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


def trajectory(bag_path, directory, cfg):
    path = directory / "trajectory.csv"
    rows = read_csv(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        statuses = [row["status"] for row in csv.DictReader(stream)]
    summary = json.loads((directory / "summary.json").read_bytes())
    check(len(rows) == summary["scans"], f"{path}: scan count differs from summary")
    check(dict(Counter(statuses)) == summary["status"], f"{path}: status counts differ from summary")
    check(statuses[-1] == summary["final_status"] and list(rows[-1][1:]) == summary["final_pose"],
          f"{path}: final status or pose differs from summary")
    with reader(bag_path) as bag:
        lidar = connections(bag, cfg)["lidar"]
        count = 0
        for conn, _, raw in bag.messages(connections=lidar):
            msg = bag.deserialize(raw, conn.msgtype)
            check_frame(msg.header.frame_id, cfg["lidar_frame"])
            t = stamp(msg) + cfg.get("header_offset_s", {}).get("lidar", 0.)
            check(count < len(rows) and rows[count].t == t, f"{path}: source scan {count+1} timestamp {t} differs")
            count += 1
        check(count == len(rows), f"{path}: {count} source scans for {len(rows)} rows")
    return rows, statuses


def check_map(grid, pixels, meta):
    info = grid.info
    origin = info.origin
    check((info.height, info.width) == pixels.shape and info.resolution == np.float32(meta["resolution"]),
          "/map: dimensions or resolution differ from map.yaml/PGM")
    check([origin.position.x, origin.position.y, origin.position.z] == [*meta["origin"][:2], 0.] and
          [origin.orientation.x, origin.orientation.y, origin.orientation.z, origin.orientation.w] == [0, 0, 0, 1],
          "/map: origin differs from map.yaml")
    probabilities = 1. - np.flipud(pixels).ravel() / 255.
    expected = np.where(probabilities > meta["occupied_thresh"], 100,
                        np.where(probabilities < meta["free_thresh"], 0, -1))
    check(np.array_equal(grid.data, expected), "/map: occupancy differs from PGM")


def audit(mapping, inspection, solution):
    cfg = load_config(solution/"profiles/mapping.json")
    mapped, _ = trajectory(mapping, solution/"map", cfg)
    cfg = load_config(solution/"profiles/inspection.json")
    inspected, statuses = trajectory(inspection, solution/"inspection", cfg)
    route = solution/"inspection/trajectory.csv"
    if cfg["odom"] is not None:
        route = route.with_name("continuous.csv")
        with route.open(encoding="utf-8-sig", newline="") as stream:
            dense_status = [row["status"] for row in csv.DictReader(stream)]
        anchors = [row for row, s in zip(read_csv(route), dense_status, strict=True) if s == "matched"]
        check(anchors == [r for r, s in zip(inspected, statuses, strict=True) if s == "matched"],
              "inspection/trajectory.csv: matched poses differ from continuous.csv anchors")
    expected = _read_poses(route, get_typestore(Stores.ROS2_HUMBLE).types, "map")
    pixels, meta, _ = read_map(solution/"map/map.yaml")
    seen = Counter()
    types = {"/map": "nav_msgs/msg/OccupancyGrid", "/pose": "geometry_msgs/msg/PoseStamped",
             "/path": "nav_msgs/msg/Path"}
    with reader(solution/"rviz") as bag:
        for c, t, raw in bag.messages():
            topic = c.topic
            if topic not in types:
                continue
            check(c.msgtype == types[topic], f"{topic}: unexpected message type")
            msg = bag.deserialize(raw, c.msgtype)
            if topic == "/map":
                check(t == expected[0][0] and astuple(msg.header) == astuple(expected[0][1].header) and
                      msg.info.map_load_time == msg.header.stamp, "/map: wrong timestamp or frame")
                check_map(msg, pixels, meta)
            elif topic == "/pose":
                i = seen[topic]
                check(i < len(expected) and t == expected[i][0] and astuple(msg) == astuple(expected[i][1]), f"/pose: sample {i+1} differs from {route.name}")
            else:
                check(t == expected[-1][0] and astuple(msg.header) == astuple(expected[-1][1].header) and
                      len(msg.poses) == len(expected), "/path: count, timestamp or frame differs")
                for i, (actual, (_, wanted)) in enumerate(zip(msg.poses, expected, strict=True)):
                    check(astuple(actual) == astuple(wanted), f"/path: sample {i+1} differs from {route.name}")
            seen[topic] += 1
    check(seen == {"/map": 1, "/pose": len(expected), "/path": 1}, f"RViz topic counts differ: {dict(seen)}")
    return dict(mapping_scans=len(mapped), inspection_scans=len(inspected), exported_route=route.name,
                rviz_poses=len(expected), first_ns=expected[0][0], last_ns=expected[-1][0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mapping", "inspection", "solution"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.mapping, args.inspection, args.solution)))
    except (AnyReaderError, OSError, KeyError, ValueError, TypeError) as exc:
        parser.exit(2, f"{exc}\n")
