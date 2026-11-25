import json
import math
import subprocess
import sys

import pytest

from mgts_nav.metrics import Pose, evaluate, read_csv


def test_common_frame_and_relative_motion():
    truth = [Pose(0, 0, 0, 0), Pose(1, 1, 0, 0)]
    estimate = [Pose(0, 3, 0, math.pi / 2), Pose(1, 3, 1, math.pi / 2)]
    result = evaluate(estimate, truth)
    assert result["ate_xy_rmse_m"] == pytest.approx(math.sqrt(7))
    assert result["rpe_translation_rmse_m"] == pytest.approx(0)
    assert result["rpe_rotation_rmse_rad"] == 0


def test_interpolation_wraps_yaw():
    estimate = [Pose(0, 0, 0, math.radians(179)), Pose(2, 2, 0, math.radians(-179))]
    result = evaluate(estimate, [Pose(1, 1, 0, math.pi), estimate[-1]])
    assert result["coverage"] == 1
    assert result["ate_xy_rmse_m"] == 0
    assert result["rpe_rotation_rmse_rad"] == pytest.approx(0)
    assert result["absolute_yaw_rmse_rad"] == pytest.approx(0)


def test_no_extrapolation_or_motion_across_gaps():
    estimate = [Pose(t, t, 0, 0) for t in (0, 1, 2, 8, 9)]
    truth = [Pose(t, t, 0, 0) for t in (-1, 2, 8, 8.5, 10)]
    result = evaluate(estimate, truth, max_gap_s=2)
    assert result["coverage"] == 3 / 5
    assert result["relative_pairs"] == 1
    missing = evaluate(estimate, [Pose(5, 5, 0, 0)])
    assert missing["coverage"] == 0
    assert missing["ate_xy_rmse_m"] is None
    assert missing["absolute_yaw_rmse_rad"] is None
    assert all(value is None for key, value in missing.items() if key.startswith("max_"))


def test_peak_errors_on_reference_clock():
    estimate = [Pose(10, 0, 0, math.radians(179)), Pose(12, 0, 0, math.radians(-179))]
    truth = [Pose(9, 100, 0, 0), Pose(10, 0, 0, math.radians(-171)),
             Pose(11, 3, 4, math.pi), Pose(12, 3, 4, estimate[-1].yaw)]
    result = evaluate(estimate, truth)
    assert (result["max_xy_error_m"], result["max_xy_error_t"]) == (5, 11)
    assert result["max_absolute_yaw_error_rad"] == pytest.approx(math.radians(10))
    assert result["max_absolute_yaw_error_t"] == 10


def test_constant_heading_bias_while_stationary(tmp_path):
    estimate, truth = (tmp_path / name for name in ("estimate.csv", "truth.csv"))
    for path, degrees in ((estimate, -175), (truth, 175)):
        yaw = math.radians(degrees)
        path.write_text(f"t,x,y,yaw\n0,0,0,{yaw}\n1,0,0,{yaw}\n")
    result = json.loads(subprocess.check_output([sys.executable, "-m", "mgts_nav.metrics", estimate, truth]))
    assert result["absolute_yaw_rmse_rad"] == pytest.approx(math.radians(10))
    assert result["ate_xy_rmse_m"] == result["rpe_rotation_rmse_rad"] == 0


def test_cli_accepts_extra_columns(tmp_path):
    path = tmp_path / "path.csv"
    path.write_text("t,x,y,yaw,status\n0,0,0,0,ok\n1,2,0,.4,ok\n")
    output = subprocess.check_output([sys.executable, "-m", "mgts_nav.metrics", str(path), str(path)])
    assert json.loads(output)["ate_xy_rmse_m"] == 0


def test_reject_duplicate_time(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("t,x,y,yaw\n0,0,0,0\n0,2,0,0\n")
    with pytest.raises(ValueError, match="timestamps"):
        read_csv(path)
