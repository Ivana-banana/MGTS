"""Experiment #88: bounded body-frame translation prior for lidar-only mapping."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from mgts_nav import slam
from mgts_nav.bag import load_config, scans
from mgts_nav.grid import Grid
from mgts_nav.metrics import evaluate, read_csv


def write_rows(path, rows):
    with path.open("x", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["t", "x", "y", "yaw", "status", "inlier_fraction"])
        writer.writerows((r.t, *r.pose, r.status, r.inliers) for r in rows)


def scored(trajectory, truth):
    rows = read_csv(trajectory)
    with trajectory.open(newline="") as src:
        statuses = Counter(r["status"] for r in csv.DictReader(src))
    metrics = evaluate(rows, truth)
    return dict(status=statuses, ate_m=metrics["ate_xy_rmse_m"], max_xy_m=metrics["max_xy_error_m"],
                final_xy_m=float(np.linalg.norm(np.array(rows[-1][1:3])-truth[-1][1:3])),
                yaw_rmse_deg=float(np.degrees(metrics["absolute_yaw_rmse_rad"])))


def experiment(data, baseline, output):
    output.mkdir(parents=True)
    profile = data/"profile.json"
    grid = Grid(.08)
    prior = slam.VelocityPrior()
    rows = list(slam.track(scans(data/"mapping", load_config(profile)), np.zeros(3), grid=grid, velocity_prior=prior))
    if len(rows) != 300:
        raise ValueError("Expected 300 mapping scans")
    grid.save(output)
    write_rows(output/"trajectory.csv", rows)
    maps = {"zero": baseline, "velocity": output}
    result = {}
    for mode, directory in maps.items():
        cmd = [sys.executable, "-m", "mgts_nav", "localize", data/"inspection", "--map", directory/"map.yaml",
               "--config", profile, "--start-search", "delayed", "--out", output/f"{mode}-blind"]
        trial = subprocess.run(cmd, capture_output=True, text=True)
        if trial.returncode not in (0, 2):
            raise RuntimeError(trial.stderr)
        result[mode] = dict(delayed_code=trial.returncode, delayed_reason=trial.stderr.strip())
    with (data/"inspection-reference.csv").open(newline="") as src:
        initial = next(csv.DictReader(src))
    for mode, directory in maps.items():
        subprocess.run([sys.executable, "-m", "mgts_nav", "localize", data/"inspection", "--map", directory/"map.yaml",
                        "--config", profile, "--initial", *(initial[k] for k in ("x", "y", "yaw")),
                        "--out", output/f"{mode}-known"], check=True)
    mapping = read_csv(data/"mapping-reference.csv")
    inspection = read_csv(data/"inspection-reference.csv")
    for mode, directory in maps.items():
        result[mode]["mapping"] = scored(directory/"trajectory.csv", mapping)
        result[mode]["known"] = scored(output/f"{mode}-known"/"trajectory.csv", inspection)
        if result[mode]["delayed_code"] == 0:
            result[mode]["blind"] = scored(output/f"{mode}-blind"/"trajectory.csv", inspection)
    result["velocity"]["prior_steps"] = prior.applied
    (output/"scores.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("baseline", type=Path, help="Map directory from the standard 300-scan run")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    experiment(args.data, args.baseline, args.output)
