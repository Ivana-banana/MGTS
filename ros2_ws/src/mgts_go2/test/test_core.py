import math
from types import SimpleNamespace

import numpy as np
import pytest

from mgts_go2.core import cloud, mapio, se3
from mgts_go2.core.deskew import deskew
from mgts_go2.core.icp import PointMap, prior_information, register
from mgts_go2.core.odom_buffer import OdomBuffer

from synthetic import base_pose, corridor, room, sweep


def test_quaternion_and_log_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        r = se3.so3_exp(rng.normal(0, 1.0, 3))
        assert np.allclose(se3.quat_to_rot(se3.rot_to_quat(r)), r, atol=1e-9)
        w = rng.normal(0, 0.8, 3)
        assert np.allclose(se3.so3_log(se3.so3_exp(w)), w, atol=1e-9)


def test_interpolate_endpoints_and_midpoint():
    a = se3.make(se3.rot_z(0.0), (0, 0, 0))
    b = se3.make(se3.rot_z(1.0), (2, 0, 0))
    assert np.allclose(se3.interpolate(a, b, 0), a)
    assert np.allclose(se3.interpolate(a, b, 1), b)
    mid = se3.interpolate(a, b, 0.5)
    assert math.isclose(se3.yaw_of(mid), 0.5, abs_tol=1e-9)
    assert np.allclose(mid[:3, 3], (1, 0, 0))


def test_planar_keeps_x_y_yaw_only():
    t = se3.make(se3.rpy_to_rot(0.1, -0.2, 0.7), (1.0, 2.0, 3.0))
    p = se3.planar(t)
    assert np.allclose(p[:3, 3], (1.0, 2.0, 0.0))
    assert math.isclose(se3.yaw_of(p), se3.yaw_of(t), abs_tol=1e-12)
    assert np.allclose(p[2, :3], (0, 0, 1))


def _fake_cloud(rows):
    dt = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'time'], 'formats': ['<f4'] * 5,
                   'offsets': [0, 4, 8, 16, 24], 'itemsize': 32})
    arr = np.zeros(len(rows), dtype=dt)
    for i, r in enumerate(rows):
        arr[i] = r
    f = lambda name, off: SimpleNamespace(name=name, offset=off, datatype=7, count=1)
    return SimpleNamespace(fields=[f('x', 0), f('y', 4), f('z', 8), f('intensity', 16), f('time', 24)],
                           point_step=32, width=len(rows), height=1, is_bigendian=False, data=arr.tobytes())


def test_decode_reads_fields_and_mask_drops_padding():
    msg = _fake_cloud([(1, 2, 3, 10, 0.01), (0, 0, 0, 0, 0), (0.1, 0, 0, 5, 0.02), (np.nan, 0, 0, 1, 0)])
    f = cloud.decode(msg)
    pts = cloud.xyz(f)
    assert np.allclose(pts[0], (1, 2, 3)) and math.isclose(f['time'][2], 0.02, rel_tol=1e-6)
    m = cloud.valid_mask(pts, min_range=0.3)
    assert m.tolist() == [True, False, False, False]  # padding zero, too close, NaN removed


def test_voxel_downsample_centroids():
    pts = np.array([[0.01, 0.01, 0.01], [0.03, 0.03, 0.03], [1.01, 0, 0]])
    out = cloud.voxel_downsample(pts, 0.1)
    assert len(out) == 2
    assert any(np.allclose(o, (0.02, 0.02, 0.02)) for o in out)


def test_odom_buffer_interpolation_and_limits():
    b = OdomBuffer()
    b.add(10.0, se3.make(None, (0, 0, 0)))
    b.add(11.0, se3.make(None, (1, 0, 0)))
    assert not b.add(10.5, se3.make(None, (9, 9, 9)))  # out of order is ignored
    assert np.allclose(b.at(10.25)[:3, 3], (0.25, 0, 0))
    assert b.at(11.05, max_extrapolation=0.1) is not None
    assert b.at(11.5, max_extrapolation=0.1) is None


def test_deskew_removes_motion_distortion():
    # robot moves at 1 m/s along x during a 0.1 s sweep; a wall point at x=5 in the world
    b = OdomBuffer()
    for k in range(0, 21):
        t = 100.0 + 0.01 * k
        b.add(t, se3.make(None, (t - 100.0, 0, 0)))
    times = np.linspace(0, 0.1, 11)
    world = np.array([5.0, 0.0, 1.0])
    raw = np.array([world - (100.0 + tt - 100.0) * np.array([1, 0, 0]) for tt in times])  # seen from moving base
    bins = 11
    fixed = deskew(raw, times, 100.0, b, bins=bins)
    # points share the pose of their time bin: error up to half a bin times the speed
    tol = 0.5 * (times.max() - times.min()) / bins * 1.0 + 1e-6
    assert np.abs(raw - world).max() > 0.09               # distortion before: 10 cm
    assert np.abs(fixed - world).max() <= tol              # after: within the binning bound


def test_icp_recovers_known_offset():
    world = room()
    target = PointMap(cloud.voxel_downsample(world, 0.05))
    true = base_pose(4.0, 3.5, 0.6)
    src = cloud.voxel_downsample(sweep(world, true, seed=2), 0.1)
    init = true @ se3.make(se3.rot_z(math.radians(4)), (0.25, -0.2, 0.0))
    res = register(src, target, init)
    dt, dr = se3.delta_norm(se3.inv(true) @ res.transform)
    assert dt < 0.02 and dr < math.radians(0.5), (dt, math.degrees(dr))
    assert res.inlier_ratio > 0.9


def test_prior_keeps_unobservable_corridor_axis():
    world = corridor()
    target = PointMap(cloud.voxel_downsample(world, 0.05))
    true = base_pose(15.0, 1.0, 0.0)
    src = cloud.voxel_downsample(sweep(world, true, seed=3), 0.1)
    init = true @ se3.make(None, (0.3, 0.1, 0.0))   # 0.3 m along the corridor, 0.1 m across
    lam = prior_information(0.01, 0.05)
    res = register(src, target, init, prior=lam)
    err_body = se3.inv(true) @ res.transform
    assert abs(err_body[1, 3]) < 0.02                # across: corrected by the walls
    assert abs(err_body[0, 3] - 0.3) < 0.03          # along: stays at the prediction


def test_pcd_and_pose_roundtrip(tmp_path):
    pts = np.random.default_rng(0).normal(size=(100, 3))
    inten = np.arange(100, dtype=float)
    mapio.write_pcd(tmp_path / 'm.pcd', pts, inten)
    p2, i2 = mapio.read_pcd(tmp_path / 'm.pcd')
    assert np.allclose(p2, pts, atol=1e-6) and np.allclose(i2, inten)
    poses = [base_pose(1, 2, 0.3), base_pose(-1, 0.5, -2.0)]
    mapio.write_poses(tmp_path / 'p.csv', [1.5, 2.5], poses)
    st, back = mapio.read_poses(tmp_path / 'p.csv')
    assert np.allclose(st, [1.5, 2.5]) and all(np.allclose(a, b, atol=1e-6) for a, b in zip(poses, back))


@pytest.mark.parametrize('field', ['x', 'y', 'z'])
def test_decode_requires_xyz(field):
    msg = _fake_cloud([(1, 2, 3, 0, 0)])
    msg.fields = [f for f in msg.fields if f.name != field]
    with pytest.raises(ValueError):
        cloud.decode(msg)
