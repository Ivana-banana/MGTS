"""Evaluate raw and gyro-corrected inspections on both velocity-prior survey maps."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from mgts_nav.metrics import read_csv
from tools.heap1_velocity_prior import scored


def experiment(data, gyro_bag, maps, output):
    output.mkdir(parents=True)
    results = {}
    commands = {}
    for survey in ("prior", "both"):
        for inspection, bag in (("raw", data / "inspection"), ("gyro", gyro_bag)):
            case = f"{survey}-{inspection}"
            cmd = [sys.executable, "-m", "mgts_nav", "localize", bag, "--config", data / "profile.json",
                   "--map", maps / survey / "map/map.yaml"]
            commands[case] = cmd
            trial = subprocess.run([*cmd, "--start-search", "delayed", "--out", output / case / "blind"],
                                   capture_output=True, text=True)
            result = {"delayed_code": trial.returncode}
            if trial.returncode:
                prefix = "Start search refused: "
                if trial.returncode != 2 or not trial.stderr.startswith(prefix):
                    raise RuntimeError(trial.stderr)
                result["refusal"] = json.loads(trial.stderr[len(prefix):])
            results[case] = result

    truth = read_csv(data / "inspection-reference.csv")
    if len(truth) != 60:
        raise ValueError("Expected 60 inspection timestamps")
    for case, cmd in commands.items():
        dest = output / case
        subprocess.run([*cmd, "--initial", *map(str, truth[0][1:]), "--out", dest / "known"], check=True)
        results[case]["known"] = scored(dest / "known/trajectory.csv", truth)
        if results[case]["delayed_code"] == 0:
            results[case]["blind"] = scored(dest / "blind/trajectory.csv", truth)
    (output / "scores.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "gyro_bag", "maps", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    experiment(args.data, args.gyro_bag, args.maps, args.output)
