import numpy as np
from scipy.spatial.transform import Rotation

from mgts_nav import localize3d


def test_deskew_uses_the_acquisition_time_of_each_point():
    times = np.array([1795604329., 1795604330.])
    motion = localize3d.Motion(times, [[0, 0, 0], [2, 0, 0]], [[0, 0, 0, 1]] * 2)
    points = np.array([[4., 0, 0], [3., 0, 0]])
    scan = motion.deskew(float(times[0]), points, np.array([0., 0.5], dtype=np.float32))
    np.testing.assert_allclose(scan, [[4, 0, 0], [4, 0, 0]])


def test_registration_recovers_a_pose_from_three_surfaces():
    a, b = np.meshgrid(np.linspace(0.3, 2.1, 21), np.linspace(0.3, 2.1, 21))
    a, b = a.ravel(), b.ravel()
    zero = np.zeros_like(a)
    walls = np.vstack((np.c_[zero, a, b], np.c_[a, zero, b], np.c_[a, b, zero]))
    shift = np.array([0.25, 0.18, 0.12])
    rotation = Rotation.from_euler('xyz', [0.02, -0.06, 0.09])
    points = rotation.inv().apply(walls - shift)
    expected = localize3d.Pose(rotation, shift)
    initial = localize3d.Pose(Rotation.identity(), shift + [0.10, -0.07, 0.05])
    fit, accepted = localize3d.SurfaceMap(walls).align(points, initial)
    assert accepted
    np.testing.assert_allclose(fit.xyz, expected.xyz, atol=0.01)
    assert (fit.rotation.inv() * expected.rotation).magnitude() < 0.01


def test_parallel_walls_leave_motion_along_the_corridor_unconstrained():
    y, z = np.meshgrid(np.linspace(0, 5, 51), np.linspace(0, 2, 21))
    y, z = y.ravel(), z.ravel()
    walls = np.vstack((np.c_[np.zeros_like(y), y, z], np.c_[np.full_like(y, 2), y, z]))
    scan = walls[::3] - [0.15, 0, 0]
    initial = localize3d.Pose(Rotation.identity(), np.array([0., 0.20, 0.]))
    fit, accepted = localize3d.SurfaceMap(walls).align(scan, initial)
    assert accepted
    assert abs(fit.xyz[0] - 0.15) < 0.01
    assert abs(fit.xyz[1] - 0.20) < 0.001


def test_registration_uses_a_nearby_plane_when_nearest_voxels_form_a_line():
    y, z = np.meshgrid(np.linspace(0, 3, 61), np.linspace(0, 2, 41))
    wall = np.c_[np.zeros(y.size), y.ravel(), z.ravel()]
    line_y = np.linspace(0.5, 2.5, 201)
    line = np.c_[np.full(len(line_y), 0.25), line_y, np.ones(len(line_y))]
    points = line[20:-20] - [0.05, 0, 0]
    initial = localize3d.Pose(Rotation.identity(), np.zeros(3))
    fit, accepted = localize3d.SurfaceMap(np.vstack((wall, line))).align(points, initial)
    assert accepted
    np.testing.assert_allclose(fit.apply(points)[:, 0], 0., atol=0.01)


def test_pose_correction_holds_while_registration_is_unavailable():
    filt = localize3d.CorrectionFilter()
    identity = Rotation.identity()
    start = localize3d.Pose(identity, np.zeros(3))
    filt.update(0., start, start, True)
    corrected = filt.update(0.07, start, localize3d.Pose(identity, np.array([0.6, 0, 0])), True)
    assert 0 < corrected.xyz[0] < 0.15
    next_odom = localize3d.Pose(identity, np.array([0.05, 0, 0]))
    predicted = filt.update(0.14, next_odom, next_odom, False)
    np.testing.assert_allclose(predicted.xyz - corrected.xyz, [0.05, 0, 0])
