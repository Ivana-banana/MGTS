import numpy as np
from scipy.spatial.transform import Rotation

from tools.tum_depth_motion import estimate


def test_pose_maps_second_camera_points_to_first_despite_bad_tracks():
    rng = np.random.default_rng(42)
    points = rng.uniform([-1., -.5, 2.], [1., .5, 4.], size=(30, 3))
    expected = Rotation.from_euler('y', 7, degrees=True).as_matrix()
    step = np.array([.03, -.01, .02])
    second = (points-step) @ expected
    second[:6] += 1
    rot, t, inliers = estimate(second, points)
    np.testing.assert_allclose(rot, expected, atol=1e-12)
    np.testing.assert_allclose(t, step, atol=1e-12)
    assert inliers == 24


def test_collinear_features_do_not_constrain_a_pose():
    xyz = np.repeat(np.arange(20.)[:, None], 3, axis=1)
    assert estimate(xyz, xyz) is None
