import json
from contextlib import nullcontext
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mgts_nav import bag


def header(ms, frame):
    sec, ns = divmod(ms * 1_000_000, 10**9)
    return NS(stamp=NS(sec=sec, nanosec=ns), frame_id=frame)


@pytest.mark.parametrize("problem", [None, "missing_before", "missing_after", "gap", "invalid_rate"])
def test_rotating_wall_on_corrected_clocks(monkeypatch, tmp_path, problem):
    mount = Rotation.from_euler("z", 0.6)
    imu_mount = Rotation.from_euler("x", 90, degrees=True)
    lever = np.array([.7, -.3, .2])
    cfg = dict(lidar="/lidar", lidar_frame="laser", imu="/imu", imu_frame="gyro", odom=None,
               base_from_lidar_xyzw=mount.as_quat().tolist(), base_from_lidar_xyz_m=lever.tolist(),
               base_from_imu_xyzw=imu_mount.as_quat().tolist(), stationary_s=0, max_sync_gap_s=.051,
               height_band_m=[0, 2], header_offset_s={"lidar": .25, "imu": -.125},
               point_time={"field": "dt", "unit": "ms", "motion": "gyro"})
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(cfg))
    # Fixed world wall x=4. Base rotates at 2 rad/s around its origin.
    dt_ms = np.array([-100, 0, 100])
    expected = np.array([[4., -1., 1.], [4., 0., 1.], [4., 1., 1.]])
    moving = Rotation.from_euler("z", -2*dt_ms/1000).apply(expected)
    points = np.column_stack((mount.inv().apply(moving-lever), dt_ms)).astype("<f8")
    fields = [NS(name=n, offset=8*i, datatype=8, count=1) for i, n in enumerate(("x", "y", "z", "dt"))]
    cloud = NS(header=header(750, "laser"), height=1, width=3, point_step=32, row_step=96,
               fields=fields, data=points.tobytes(), is_bigendian=False)
    imu = []
    for ms in (900, 950, 1000, 1050, 1100, 1150):
        if (problem, ms) in (("missing_before", 900), ("missing_after", 1100), ("gap", 950)):
            continue
        if problem == "missing_after" and ms > 1100:
            continue
        cov = [-1 if problem == "invalid_rate" and ms == 950 else 0] + [0]*8
        imu.append(NS(header=header(ms+125, "gyro"), angular_velocity=NS(x=0., y=2., z=0.),
                      angular_velocity_covariance=cov))
    streams = {"/lidar": [cloud], "/imu": imu}
    conns = [NS(topic="/lidar", msgtype="sensor_msgs/msg/PointCloud2"), NS(topic="/imu", msgtype="sensor_msgs/msg/Imu")]
    source = NS(connections=conns, deserialize=lambda raw, _: raw,
                messages=lambda connections: ((c, 0, m) for c in connections for m in streams[c.topic]))
    monkeypatch.setattr(bag, "reader", lambda _: nullcontext(source))
    scan, = bag.scans("fixture", bag.load_config(path))
    assert scan.t == 1.
    if problem:
        assert scan.empty_reason == "deskew_imu_unavailable" and not len(scan.xy)
    else:
        assert scan.empty_reason is None
        np.testing.assert_allclose(scan.xy[np.argsort(scan.xy[:, 1])], expected[:, :2], atol=1e-12)
        np.testing.assert_allclose(scan.origin, lever[:2])
    if problem is not None:
        return
    cfg["point_time"]["motion"] = "guess"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="point_time.motion"):
        bag.load_config(path)
    cfg["point_time"]["motion"] = "gyro"
    cfg["imu"] = None
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="IMU topic"):
        bag.load_config(path)
