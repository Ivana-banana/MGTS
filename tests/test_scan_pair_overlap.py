import numpy as np

from tools.scan_pair_overlap import scan


def test_occlusion_and_rotation_in_sensor_axes():
    pose = np.zeros(3)
    front = scan(pose, (0, 1))
    np.testing.assert_allclose(front, [[6.75, 0.]])
    hit = scan(pose, (30, 31))
    np.testing.assert_allclose(hit, [[2., 2 * np.tan(np.deg2rad(30))]])
    pose[2] = np.deg2rad(30)
    np.testing.assert_allclose(scan(pose, (0, 1)), [[2 / np.cos(pose[2]), 0.]])
