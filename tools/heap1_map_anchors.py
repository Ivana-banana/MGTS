"""Compare known-start tracking and blind delayed anchors on two HEAP-1 maps."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from mgts_nav.bag import load_config, scans
from mgts_nav.delayed_start import estimate_start
from mgts_nav.grid import read_map
from mgts_nav.metrics import evaluate, read_csv, wrap


def compare(data, short_map, long_map, output):
    output.mkdir(parents=True)
    profile = data/"profile.json"
    stream = list(scans(data/"inspection", load_config(profile)))
    report = {}
    maps = {"short": short_map, "long": long_map}
    for name, path in maps.items():
        pixels, meta, _ = read_map(path)
        stats = {}
        try:
            stats["start"] = estimate_start(stream, pixels, meta, stats).tolist()
        except ValueError as exc:
            stats["rejection"] = str(exc)
        report[name] = dict(delayed=stats)

    reference = data/"inspection-reference.csv"
    with reference.open(newline="") as src:
        first = next(csv.DictReader(src))
    for name, path in maps.items():
        dest = output/name
        subprocess.run([sys.executable, "-m", "mgts_nav", "localize", data/"inspection", "--map", path,
                        "--config", profile, "--initial", *(first[k] for k in ("x", "y", "yaw")),
                        "--out", dest], check=True)

    truth = read_csv(reference)
    if len(truth) != 60 or len(stream) != len(truth):
        raise ValueError("Expected 60 inspection samples")
    np.testing.assert_allclose([s.t for s in stream], [r.t for r in truth], atol=1e-6, rtol=0)
    for name, row in report.items():
        trajectory = output/name/"trajectory.csv"
        metrics = evaluate(read_csv(trajectory), truth)
        with trajectory.open(newline="") as src:
            matches = [float(r["inlier_fraction"]) for r in csv.DictReader(src) if r["status"] == "matched"]
        row["known_start"] = dict(matched=len(matches), metrics=metrics, support_median=float(np.median(matches)),
                                  support_min=min(matches)) if matches else dict(matched=0, metrics=metrics)
        for anchor in row["delayed"]["anchors"]:
            ref = truth[anchor["index"]]
            anchor["reference"] = [ref.x, ref.y, ref.yaw]
            if anchor["candidate"] is not None:
                x, y, yaw = anchor["candidate"]
                anchor["delta"] = [x-ref.x, y-ref.y, float(np.degrees(wrap(yaw-ref.yaw)))]
    (output/"scores.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "short_map", "long_map", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    compare(args.data, args.short_map, args.long_map, args.output)
