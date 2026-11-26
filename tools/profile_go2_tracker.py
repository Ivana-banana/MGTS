"""Measure mutually exclusive costs of the Go2 3D tracker on an inspection recording."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from mgts_nav.bag import cloud_field, cloud_xyz, reader, stamp
from mgts_nav.localize3d import CorrectionFilter, Motion, SurfaceMap, odometry_sample, scan_points


class TimedTree:
    def __init__(self, tree):
        self.tree = tree
        self.data = tree.data
        self.n = tree.n
        self.elapsed_ns = 0

    def query(self, *args, **kwargs):
        start = perf_counter_ns()
        result = self.tree.query(*args, **kwargs)
        self.elapsed_ns += perf_counter_ns() - start
        return result


def profile(path, map_file, csv_file, native=None):
    with np.load(map_file) as data:
        surface = SurfaceMap(data["xyz"])
    if native is None:
        surface.tree = TimedTree(surface.tree)
        surface.plane_tree = TimedTree(surface.plane_tree)
    else:
        # The C++ core does its own neighbour search: neighbors_ms stays 0, align is in rest_of_align_ms.
        from mgts_nav.native_surface import NativeSurface, load_dll
        surface = NativeSurface(surface, load_dll(native))
    filt = CorrectionFilter()
    rows = []
    with reader(path) as bag:
        conns = {c.topic: c for c in bag.connections}
        odom = conns["/utlidar/robot_odom"]
        samples = []
        for c, _, raw in bag.messages(connections=[odom]):
            msg = bag.deserialize(raw, c.msgtype)
            samples.append((stamp(msg), *odometry_sample(msg)))
        motion = Motion(*zip(*samples))
        for c, _, raw in bag.messages(connections=[conns["/utlidar/cloud_base"]]):
            start = perf_counter_ns()
            msg = bag.deserialize(raw, c.msgtype)
            t = stamp(msg)
            xyz = cloud_xyz(msg)
            offsets = cloud_field(msg, "time", {7: "f4"})
            decoded = perf_counter_ns()
            xyz, offsets = scan_points(xyz, offsets)
            if not len(xyz) or not motion.covers(t + min(0., offsets.min().item()),
                                                 t + max(0., offsets.max().item())):
                continue
            filtered = perf_counter_ns()
            points = motion.deskew(t, xyz, offsets)
            prior = motion.at(t)
            selected = points[np.arange(len(points)) % 4 != 0]
            ready = perf_counter_ns()
            if native is None:
                surface.tree.elapsed_ns = surface.plane_tree.elapsed_ns = 0
            raw_pose, accepted = surface.align(selected, prior)
            aligned = perf_counter_ns()
            pose = filt.update(t, prior, raw_pose, accepted)
            end = perf_counter_ns()
            query = 0 if native else surface.tree.elapsed_ns + surface.plane_tree.elapsed_ns
            costs = np.array([decoded-start, filtered-decoded, ready-filtered, query,
                              aligned-ready-query, end-aligned, end-start]) / 1e6
            rows.append([t, *pose.xyz, *pose.rotation.as_quat(), accepted, *costs])
    if not rows:
        raise ValueError(f"{path}: no scans with odometry")
    names = ["decode_ms", "filter_ms", "deskew_prior_ms", "neighbors_ms",
             "rest_of_align_ms", "correction_ms", "total_ms"]
    with csv_file.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["t", "x", "y", "z", "qx", "qy", "qz", "qw", "accepted", *names])
        writer.writerows(rows)
    costs = np.array([r[-len(names):] for r in rows])
    return {"scans": len(rows), "matched": sum(r[8] for r in rows),
            "mean_ms": dict(zip(names, costs.mean(axis=0))),
            "p95_ms": dict(zip(names, np.percentile(costs, 95, axis=0)))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("map_file", type=Path)
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--native", type=Path, help="DLL built from tools/native_surface.cpp instead of SciPy")
    print(json.dumps(profile(**vars(parser.parse_args())), indent=2))
