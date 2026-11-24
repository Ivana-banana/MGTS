"""Reproduce the two corridor trials with pytest -q -s tests/test_corridor.py."""
from dataclasses import replace
import json
import subprocess
import sys

import numpy as np
import pytest

from mgts_nav.bag import load_config, scans
from mgts_nav.demo import PROFILE
from mgts_nav.grid import Grid, read_map
from mgts_nav.metrics import Pose, evaluate, read_csv
from mgts_nav.slam import track


def cli(*args):
    return subprocess.run([sys.executable, "-m", *map(str, args)], capture_output=True, text=True)


@pytest.mark.parametrize("seed", [41, 73])
def test_corridor(tmp_path, seed):
    demo = tmp_path / "demo"
    out = tmp_path / "map"
    cli("mgts_nav.demo", demo, "--layout", "corridor", "--seed", seed).check_returncode()
    common = ["--config", PROFILE]
    cli("mgts_nav", "map", demo / "mapping", *common, "--out", out).check_returncode()
    initial = json.loads((demo / "initial_pose.json").read_text())
    map_truth = read_csv(demo / "mapping_truth.csv")
    truth = read_csv(demo / "inspection_truth.csv")
    results = {}

    def score(name, poses, reference):
        metric = evaluate(poses, reference)
        assert metric["coverage"] == 1
        results[name] = metric["ate_xy_rmse_m"]

    score("mapping_all", read_csv(out / "trajectory.csv"), map_truth)
    points = read_map(out / "map.yaml")[2]
    cfg = load_config(PROFILE)
    grid = Grid(0.08)
    for stage, ref in [("mapping", map_truth), ("inspection", truth)]:
        lidar = (replace(s, odom=None, gyro_delta=None) for s in scans(demo / stage, cfg))
        if stage == "mapping":
            rows = list(track(lidar, np.zeros(3), grid=grid))
        else:
            rows = list(track(lidar, initial, map_points=points))
        score(f"{stage}_lidar_only", [Pose(r.t, *r.pose) for r in rows], ref)
        results[f"{stage}_lidar_prediction_only"] = sum(r.status == "prediction_only" for r in rows)

    for mode in ("known", "automatic"):
        loc = tmp_path / mode
        start = ["--initial", *initial] if mode == "known" else []
        run = cli("mgts_nav", "localize", demo / "inspection", "--map", out / "map.yaml",
                  *common, *start, "--out", loc)
        if mode == "automatic" and run.returncode:
            assert "Provide --initial" in run.stderr, run.stderr
            results["auto_refusal"] = run.stderr.strip()
            continue
        run.check_returncode()
        summary = json.loads((loc / "summary.json").read_text())
        score(mode, read_csv(loc / "trajectory.csv"), truth)
        results[f"{mode}_matched"] = summary["status"].get("matched", 0)
        results[f"{mode}_start"] = summary["initial_pose"]
        base = tmp_path / f"baseline-{mode}"
        cli("mgts_nav", "baseline", demo / "inspection", *common,
            "--initial", *summary["initial_pose"], "--out", base).check_returncode()
        score(f"kinematic_{mode}", read_csv(base / "trajectory.csv"), truth)
    assert results["known"] < results["kinematic_known"]
    print(json.dumps({"seed": seed, **results}, allow_nan=False))
