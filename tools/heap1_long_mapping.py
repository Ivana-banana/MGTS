"""Run and score the fixed 300-scan HEAP-1 mapping experiment."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

from mgts_nav.metrics import read_csv


def score(rows, truth, statuses):
    estimates = np.asarray(rows)
    reference = np.asarray(truth)
    np.testing.assert_allclose(estimates[:, 0], reference[:, 0], rtol=0, atol=1e-6)
    errors = estimates[:, 1:] - reference[:, 1:]
    xy = np.linalg.norm(errors[:, :2], axis=1)
    yaw = np.degrees(np.arctan2(np.sin(errors[:, 2]), np.cos(errors[:, 2])))
    def length(trajectory):
        return float(np.linalg.norm(np.diff(trajectory[:, 1:3], axis=0), axis=1).sum())
    return dict(scans=len(xy), matched=statuses.count("matched"), xy_rmse_m=float(np.sqrt(np.mean(xy**2))),
                yaw_rmse_deg=float(np.sqrt(np.mean(yaw**2))), last_xy_m=float(xy[-1]),
                last_abs_yaw_deg=float(abs(yaw[-1])), estimated_length_m=length(estimates),
                reference_length_m=length(reference), duration_s=float(reference[-1, 0]-reference[0, 0]))


def load_result(output):
    path = output / "trajectory.csv"
    with path.open(newline="") as stream:
        statuses = [row["status"] for row in csv.DictReader(stream)]
    return read_csv(path), statuses


def experiment(data, output):
    output.mkdir(parents=True)
    cli = [sys.executable, "-m", "mgts_nav"]
    profile = ["--config", str(data / "profile.json")]
    start = perf_counter()
    subprocess.run([*cli, "map", data/"mapping", *profile, "--resolution", ".08", "--out", output/"map"],
                   check=True)
    map_seconds = perf_counter()-start
    start = perf_counter()
    loc = subprocess.run([*cli, "localize", data/"inspection", *profile, "--map", output/"map/map.yaml",
                          "--start-search", "delayed", "--out", output/"inspection"], capture_output=True, text=True)
    loc_seconds = perf_counter()-start
    if loc.returncode not in (0, 2):
        raise RuntimeError(loc.stderr)
    # Open the independent reference after both predictions.
    rows, statuses = load_result(output/"map")
    truth = read_csv(data/"mapping-reference.csv")
    if len(rows) != len(truth) or len(rows) != 300:
        raise ValueError("Expected 300 mapping scans")
    report = dict(map_s=map_seconds, localize_s=loc_seconds, mapping=score(rows, truth, statuses), thirds=[])
    for i in range(0, 300, 100):
        block = slice(i, i+100)
        report["thirds"].append(dict(first_frame=150+i, **score(rows[block], truth[block], statuses[block])))
    if loc.returncode:
        report["inspection"] = dict(exit_code=loc.returncode, reason=loc.stderr.strip())
    else:
        rows, statuses = load_result(output/"inspection")
        refs = read_csv(data/"inspection-reference.csv")
        if len(rows) != 60 or len(refs) != 60:
            raise ValueError("Expected 60 inspection scans")
        report["inspection"] = score(rows, refs, statuses)
    (output/"scores.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    experiment(args.data, args.output)
