import csv
import json

import numpy as np
import pytest

from mgts_nav.bag import load_config, scans
from mgts_nav.demo import PROFILE, generate


def test_demo_bags_feed_the_navigation_adapter(tmp_path):
    root = tmp_path / "demo"
    generate(root)
    config = load_config(PROFILE)
    for run in ("mapping", "inspection"):
        with (root / f"{run}_truth.csv").open(newline="", encoding="utf-8") as f:
            truth = list(csv.DictReader(f))
        observed = list(scans(root / run, config))
        assert len(truth) == len(observed)
        assert len(observed) >= 100
        assert [s.t for s in observed] == pytest.approx([float(r["t"]) for r in truth])
        assert all(len(s.xy) > 80 for s in observed)
        assert all(s.odom is not None for s in observed)
        assert all(s.gyro_delta is not None for s in observed[1:])
        poses = np.array([[float(r[k]) for k in ("x", "y", "yaw")] for r in truth])
        assert np.allclose(poses[:8], poses[0])
        assert np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1).max() < 0.15
        assert np.abs(np.diff(poses[:, 2])).max() < 0.12
        if run == "mapping":
            assert np.allclose(poses[-1, :2], poses[0, :2])
            assert np.allclose(poses[0], 0)
        else:
            initial = json.loads((root / "initial_pose.json").read_text())
            assert np.allclose(poses[0], initial)
    with pytest.raises(FileExistsError):
        generate(root)
