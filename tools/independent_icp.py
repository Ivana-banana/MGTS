"""Compare MGTS tracking with the pinned optional trimesh ICP on identical inputs."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter, process_time
from zipfile import ZipFile

import numpy as np
import trimesh

from mgts_nav.bag import load_config, scans
from mgts_nav.coarse_start import estimate
from mgts_nav.grid import read_map
from mgts_nav.metrics import read_csv
from mgts_nav.slam import rotation, track, world
from tools.kitti_repeat import calibration, load_scan, report


def icp(scan, target_xyz, previous):
    seed = np.eye(4)
    seed[:2, :2] = rotation(previous[2])
    seed[:2, 3] = previous[:2]
    source = np.column_stack((scan, np.zeros(len(scan))))
    transform, _, _ = trimesh.registration.icp(source, target_xyz, initial=seed,
                                                threshold=1e-5, max_iterations=24,
                                                reflection=False, scale=False)
    if transform[2, 2] < .99999:
        raise ValueError("Non-planar ICP result")
    return np.r_[transform[:2, 3], np.arctan2(transform[1, 0], transform[0, 0])]


def control():
    xy = np.random.default_rng(42).normal(size=(100, 2))
    expected = np.array([.04, -.03, .025])
    target = np.column_stack((world(xy, expected), np.zeros(len(xy))))
    np.testing.assert_allclose(icp(xy, target, np.zeros(3)), expected, atol=1e-8)


def compare(stream, truth, map_file):
    stream = [replace(s, odom=None, gyro_delta=None) for s in stream]
    times = np.array([s.t for s in stream])
    np.testing.assert_allclose(times, [p.t for p in truth], atol=1e-6, rtol=0)
    truth = np.array(truth)[:, 1:]
    pixels, meta, points = read_map(map_file)
    starts = {"known": truth[0]}
    result = {}
    wall, cpu = perf_counter(), process_time()
    try:
        starts["blind"] = estimate(stream[0], pixels, meta, {})
    except ValueError as exc:
        result["blind"] = {"start_refused": str(exc), "coverage": 0}
    result["shared_search"] = dict(seconds=perf_counter()-wall, cpu_s=process_time()-cpu)
    target = np.column_stack((points, np.zeros(len(points))))
    for mode, start in starts.items():
        result[mode] = {}
        for method in ("mgts", "trimesh"):
            wall, cpu = perf_counter(), process_time()
            if method == "mgts":
                rows = list(track(stream, start, map_points=points))
                poses, statuses = [r.pose for r in rows], [r.status for r in rows]
            else:
                poses, statuses = [], []
                pose = start.copy()
                for s in stream:
                    status = "prediction_only"
                    if len(s.xy) >= 3:
                        pose = icp(s.xy, target, pose)
                        status = "registered"
                    poses.append(pose.copy())
                    statuses.append(status)
            elapsed, used = perf_counter()-wall, process_time()-cpu
            result[mode][method] = dict(report(times, truth, poses, statuses), seconds=elapsed, cpu_s=used)
    return result


def run(office, archives, field):
    if trimesh.__version__ != "4.5.3":
        raise ValueError("Requires trimesh 4.5.3")
    control()
    result = {}
    truth = read_csv(office / "data/inspection_truth.csv")
    office_scans = scans(office / "data/inspection", load_config(Path("configs/example.json")))
    result["office"] = compare(office_scans, truth, office / "map/map.yaml")
    first = json.loads((field / "scores.json").read_text())["inspection_first"]
    truth = read_csv(field / "inspection-reference.csv")
    with ZipFile(archives / "drive.zip") as drive, ZipFile(archives / "calib.zip") as calib:
        transform = calibration(calib)
        stream = [load_scan(drive, first+i, p.t, *transform) for i, p in enumerate(truth)]
    result["kitti"] = compare(stream, truth, field / "map.yaml")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("office", "archives", "field", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.office, args.archives, args.field), indent=2,
                                      allow_nan=False), encoding="utf-8")
