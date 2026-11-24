"""Run the fixed office lidar-extrinsics sensitivity experiment (issue 46)."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from scipy.spatial.transform import Rotation

from mgts_nav.demo import PROFILE, WALLS, generate
from mgts_nav.grid import read_map
from mgts_nav.map_eval import evaluate_map
from mgts_nav.metrics import evaluate, read_csv, wrap


def run(*args):
    subprocess.run([sys.executable, "-m", "mgts_nav", *map(str, args)], check=True,
                   stdout=subprocess.DEVNULL)


def profiles(directory):
    paths = {}
    for name, yaw_deg, dx in (("control", 0, 0), ("yaw", 5, 0), ("x", 0, .10), ("both", 5, .10)):
        cfg = json.loads(PROFILE.read_bytes())
        rotation = Rotation.from_euler("z", yaw_deg, degrees=True) * Rotation.from_quat(cfg["base_from_lidar_xyzw"])
        cfg["base_from_lidar_xyzw"] = rotation.as_quat().tolist()
        cfg["base_from_lidar_xyz_m"][0] += dx
        path = directory / f"profile-{name}.json"
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        paths[name] = path
    return paths


def scores(output, truth):
    summary = json.loads((output / "summary.json").read_bytes())
    estimated = read_csv(output / "trajectory.csv")
    reference = read_csv(truth)
    metrics = evaluate(estimated, reference)
    # This fixture supplies independent truth at each lidar timestamp.
    squared = []
    for row, ref in zip(estimated, reference, strict=True):
        if abs(row.t - ref.t) > 1e-6:
            raise ValueError("Estimate and reference timestamps differ")
        squared.append(wrap(row.yaw - ref.yaw)**2)
    yaw_rmse_deg = math.degrees(math.sqrt(sum(squared) / len(squared)))
    return dict(ate_m=metrics["ate_xy_rmse_m"], yaw_rmse_deg=yaw_rmse_deg, matched=summary["status"].get("matched", 0),
                scans=summary["scans"], coverage=metrics["coverage"])


def experiment(root):
    root.mkdir(parents=True)
    configs = profiles(root)
    rows = []
    for seed in (41, 73):
        directory = root / str(seed)
        data = directory / "data"
        generate(data, layout="office", seed=seed)
        for map_name, map_config in configs.items():
            output = directory / f"map-{map_name}"
            run("map", data / "mapping", "--config", map_config, "--resolution", ".08", "--out", output)
            mapping = scores(output, data / "mapping_truth.csv")
            _, meta, points = read_map(output / "map.yaml")
            geometry = evaluate_map(points, WALLS, meta["resolution"], .12)
            inspection_names = configs if map_name == "control" else [map_name]
            for name in inspection_names:
                route = directory / f"inspection-{map_name}-{name}"
                run("localize", data / "inspection", "--map", output / "map.yaml", "--config", configs[name],
                    "--initial-file", data / "initial_pose.json", "--out", route)
                inspection = scores(route, data / "inspection_truth.csv")
                rows.append(dict(seed=seed, mapping_profile=map_name, inspection_profile=name,
                                 mapping=mapping, inspection=inspection, geometry=geometry))
    (root / "results.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="Fresh output directory")
    experiment(parser.parse_args().out)
