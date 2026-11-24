"""Evaluate PR 35's fixed clock-lag estimator on three public HEAP-1 windows."""
import argparse
import csv
import importlib.util
from itertools import islice
import json
from pathlib import Path
import subprocess

import numpy as np

from mgts_nav.bag import load_config, scans
from mgts_nav.grid import Grid, read_map
from mgts_nav.metrics import read_csv
from mgts_nav.slam import track


REVISION = "997f7becc0f7806b0035e19789bc855c3e248541"
BLOB = "04d13ec3937d005219403381d3fc271b6fd8eaea"


def proposal(output):
    actual = subprocess.check_output(["git", "rev-parse", REVISION + ":tools/clock_lag.py"], text=True).strip()
    if actual != BLOB:
        raise ValueError("Unexpected PR implementation")
    path = output / "clock_lag.py"
    path.write_bytes(subprocess.check_output(["git", "cat-file", "blob", BLOB]))
    spec = importlib.util.spec_from_file_location("clock_lag_pr35", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(original, heldout, output):
    output.mkdir(parents=True)
    clock = proposal(output)
    rows = []
    windows = [(original, "mapping", 150, 40), (original, "inspection", 2560, 60),
               (heldout, "inspection", 2620, 60)]
    for data, role, first, count in windows + windows[1:]:
        cfg = load_config(data/"profile.json")
        bag = data/role
        lidar = list(islice(scans(bag, dict(cfg, imu=None, odom=None)), count))
        if len(lidar) != count or any(s.gyro_delta is not None or s.odom is not None for s in lidar):
            raise ValueError("Expected fixed-length LiDAR-only window")
        reference = read_csv(data/f"{role}-reference.csv")[:count]
        fixed_map = len(rows) >= len(windows)
        if fixed_map:
            estimates = list(track(lidar, np.array(reference[0][1:]), map_points=points))
        else:
            grid = Grid(.08)
            estimates = list(track(lidar, np.zeros(3), grid=grid))
            if role == "mapping":
                grid.save(output)
                _, _, points = read_map(output / "map.yaml")
        method = "fixed" if fixed_map else "local"
        csv_path = output / f"lidar-{first}-{method}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["t", "x", "y", "yaw", "status"])
            writer.writerows((r.t, *r.pose, r.status) for r in estimates)
        ref_t, ref_rate = clock.trajectory_rate(csv_path)
        t, rate = clock.sensor_rate(bag, cfg, "imu")
        within = (t >= lidar[0].t) & (t <= lidar[-1].t)
        t, rate = t[within], rate[within]
        if not np.allclose([r.t for r in reference], [s.t for s in lidar], atol=1e-6, rtol=0):
            raise ValueError("CPT7 and scan timestamps differ")
        row = dict(first=first, method=method, scans=count, matched=sum(r.status == "matched" for r in estimates),
                   lidar_range_deg=float(np.rad2deg(np.ptp(np.unwrap([r.pose[2] for r in estimates])))),
                   cpt7_range_deg=float(np.rad2deg(np.ptp(np.unwrap([r.yaw for r in reference])))),
                   lidar_rate_sd_deg_s=float(np.rad2deg(np.std(ref_rate))) if len(ref_rate) else None,
                   imu_rate_sd_deg_s=float(np.rad2deg(np.std(rate))), trials=[])
        for shift in (0., -.20, .20):
            if len(ref_t) < 2:
                row["reason"] = "Too few reference rate samples"
                break
            lag, correlation = clock.estimate(ref_t, ref_rate, t+shift, rate)
            row["trials"].append(dict(shift_s=shift, lag_s=lag, correlation=correlation))
        rows.append(row)
        print(json.dumps(row), flush=True)
    (output / "report.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "heldout", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    evaluate(args.original, args.heldout, args.output)
