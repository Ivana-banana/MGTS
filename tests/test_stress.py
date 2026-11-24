import hashlib

import numpy as np
import pytest

from mgts_nav.bag import load_config, scans
from mgts_nav.demo import PROFILE, generate
from mgts_nav.metrics import Pose
from mgts_nav.slam import Estimate
from mgts_nav.stress import fault_bag, summarize


def test_fault_copies_change_only_the_intended_sensor(tmp_path):
    root = tmp_path / "demo"
    generate(root)
    source = root / "inspection"
    db = next(source.glob("*.db3"))
    digest = hashlib.sha256(db.read_bytes()).digest()
    cfg = load_config(PROFILE)
    clean = list(scans(source, cfg))
    for case in ("lidar_no_returns", "lidar_no_messages", "imu_gap", "odom_drift"):
        target = tmp_path / case
        changed = fault_bag(source, target, case, cfg)
        rows = list(scans(target, cfg))
        if case == "lidar_no_messages":
            assert len(clean) - len(rows) == changed == 20
            assert max(np.diff([r.t for r in rows])) == pytest.approx(2.1)
            continue
        assert [r.t for r in rows] == [r.t for r in clean]
        if case == "lidar_no_returns":
            assert sum(len(s.xy) == 0 for s in rows) == changed == 20
        else:
            assert all(np.array_equal(s.xy, c.xy) for s, c in zip(rows, clean))
        if case == "imu_gap":
            assert changed == 100
            assert sum(s.gyro_delta is None for s in rows) > 20
        if case == "odom_drift":
            np.testing.assert_array_equal(rows[0].odom, clean[0].odom)
            assert rows[-1].odom[0] - clean[-1].odom[0] > 2
        else:
            np.testing.assert_allclose([r.odom for r in rows], [r.odom for r in clean])
    assert hashlib.sha256(db.read_bytes()).digest() == digest


def test_prediction_series_and_temporal_coverage_are_separate():
    statuses = ["matched", "prediction_only", "prediction_only", "matched", "prediction_only"]
    rows = [Estimate(t, np.zeros(3), s, 0) for t, s in zip([0, 1, 2, 8, 9], statuses)]
    result = summarize(rows, [Pose(t, 0, 0, 0) for t in range(10)])
    assert result["matched_fraction"] == 2 / 5
    assert result["truth_coverage"] == 0.5
    assert result["max_prediction_only_scans"] == 2
    assert result["max_pose_interval_s"] == 6
