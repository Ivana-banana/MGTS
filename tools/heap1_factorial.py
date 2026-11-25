"""Compare four HEAP-1 survey maps on one unchanged inspection (#96)."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from mgts_nav.metrics import read_csv
from tools.heap1_drift import rotate
from tools.heap1_velocity_prior import scored


def experiment(data, gyro_bag, output):
    output.mkdir(parents=True)
    cli = [sys.executable, "-m", "mgts_nav"]
    profile = ["--config", data / "profile.json"]
    results = {}
    for mode, bag, flags in (
        ("raw", data / "mapping", []),
        ("prior", data / "mapping", ["--mapping-velocity-prior"]),
        ("gyro", gyro_bag, []),
        ("both", gyro_bag, ["--mapping-velocity-prior"]),
    ):
        directory = output / mode
        subprocess.run([*cli, "map", bag, *profile, "--resolution", ".08", *flags,
                        "--out", directory / "map"], check=True)
        localize = [*cli, "localize", data / "inspection", *profile, "--map", directory / "map/map.yaml"]
        trial = subprocess.run([*localize, "--start-search", "delayed", "--out", directory / "blind"],
                               capture_output=True, text=True)
        if trial.returncode != 0 and not (trial.returncode == 2 and
                                          trial.stderr.strip().startswith("Start search refused:")):
            raise RuntimeError(f"{mode}: {trial.stderr}")
        results[mode] = dict(delayed_code=trial.returncode, delayed_reason=trial.stderr.strip())

    # Open references only after all map and blind runs have finished.
    mapping = read_csv(data / "mapping-reference.csv")
    inspection = read_csv(data / "inspection-reference.csv")
    if len(mapping) != 300 or len(inspection) != 60:
        raise ValueError("Unexpected HEAP-1 reference window")
    for mode, result in results.items():
        directory = output / mode
        subprocess.run([*cli, "localize", data / "inspection", *profile, "--map", directory / "map/map.yaml",
                        "--initial", *map(str, inspection[0][1:]), "--out", directory / "known"], check=True)
        trajectory = directory / "map/trajectory.csv"
        rows, truth = np.asarray(read_csv(trajectory)), np.asarray(mapping)
        np.testing.assert_allclose(rows[:, 0], truth[:, 0], rtol=0, atol=1e-6)
        result["mapping"] = scored(trajectory, mapping)
        # Source frames 220 and 230 are rows 70 and 80 in the 150..449 survey.
        def relative_xy(p):
            return rotate(p[80:81, 1:3] - p[70:71, 1:3], -p[70:71, 3])
        result["error_220_230_m"] = float(np.linalg.norm(relative_xy(rows) - relative_xy(truth)))
        result["known"] = scored(directory / "known/trajectory.csv", inspection)
        if result["delayed_code"] == 0:
            result["blind"] = scored(directory / "blind/trajectory.csv", inspection)
    (output / "scores.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("gyro_bag", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    experiment(args.data, args.gyro_bag, args.output)
