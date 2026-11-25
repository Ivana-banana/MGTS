"""Replay four Intel Lab configurations on the fixed #107 time windows."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from mgts_nav.bag import interval_samples
from mgts_nav.metrics import read_csv
from tools.intel_lab import laser_rows, odom_rows, write_window
from tools.intel_laser_clock import analyze


def wheel_coverage(rows, times, gap):
    stamps = np.array([ns for ns, _ in rows]) / 1e9
    covered = sum(interval_samples(times, t, t, gap) is not None for t in stamps)
    breaks = []
    for i, (a, b) in enumerate(zip(stamps, stamps[1:])):
        if b-a > .35:
            breaks.append(dict(index=i, duration_s=b-a, wheel=interval_samples(times, a, b, gap) is not None))
    return dict(scans=len(rows), wheel_covered=covered, breaks=breaks)


def compare_epochs(csv_path, reference):
    rows = np.array(read_csv(csv_path))
    errors = []
    for t, x, y, _ in reference:
        matches = rows[np.abs(rows[:, 0]-t) <= 1e-6]
        if len(matches) != 1:
            raise ValueError(f"Unpaired external epoch {t}")
        errors.append(np.linalg.norm(matches[0, 1:3]-[x,y]))
    return dict(epochs=len(errors), rms_xy_m=float(np.sqrt(np.mean(np.square(errors)))))


def experiment(source, data, output):
    output.mkdir(parents=True)
    raw = list(laser_rows(source))
    wheels = odom_rows(source)
    times = np.array([t for t, _ in wheels])/1e9
    profile = json.loads((data/"profile.json").read_text())
    if profile["odom"] != "/odom":
        raise ValueError("Expected the wheel-odometry profile prepared for #108")
    windows = {}
    report = {"coverage": {}, "trials": {}}
    for name, sl in (("mapping", slice(169,1112)), ("inspection", slice(3420,3498))):
        audit, control, ordered = analyze(raw[sl])
        if audit["conflicting_groups"] or control[0][0] != ordered[0][0] or control[-1][0] != ordered[-1][0]:
            raise ValueError("Incompatible Intel Lab time windows")
        windows[name] = (control, ordered)
        report["coverage"][name] = {
            label: wheel_coverage(rows, times, profile["max_sync_gap_s"])
            for label, rows in zip(("control", "sorted"), windows[name])
        }
    for sorted_ipc, order in enumerate(("control", "sorted")):
        for use_wheel in (False, True):
            mode = order + ("-odom" if use_wheel else "-lidar")
            run = output/mode
            run.mkdir()
            cfg = profile if use_wheel else dict(profile, odom=None)
            config = run/"profile.json"
            config.write_text(json.dumps(cfg))
            for name, versions in windows.items():
                write_window(run/name, versions[sorted_ipc], wheels if use_wheel else None)
            cli = [sys.executable, "-m", "mgts_nav"]
            subprocess.run([*cli, "map", run/"mapping", "--config", config, "--resolution", ".08",
                            "--out", run/"map"], check=True, stdout=subprocess.DEVNULL)
            loc = [*cli, "localize", run/"inspection", "--map", run/"map/map.yaml", "--config", config]
            blind = subprocess.run([*loc, "--start-search", "delayed", "--out", run/"blind"],
                                   capture_output=True, text=True)
            result = dict(delayed_code=blind.returncode)
            if blind.returncode:
                prefix = "Start search refused: "
                if blind.returncode != 2 or not blind.stderr.startswith(prefix):
                    raise RuntimeError(blind.stderr)
                result["refusal"] = json.loads(blind.stderr[len(prefix):])
            subprocess.run([*loc, "--initial-file", data/"external-start.json", "--out", run/"known"],
                           check=True, stdout=subprocess.DEVNULL)
            for name, product in (("mapping","map"), ("inspection","known")):
                summary = json.loads((run/product/"summary.json").read_text())
                result[name] = dict(status=summary["status"], **compare_epochs(
                    run/product/"trajectory.csv", read_csv(data/f"{name}-external.csv")))
            report["trials"][mode] = result
            print(mode, json.dumps(result), flush=True)
    (output/"scores.json").write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("data", type=Path, help="Intel Lab data directory generated with --with-odom (#108)")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    experiment(args.source, args.data, args.output)
