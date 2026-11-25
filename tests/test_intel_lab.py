import numpy as np
from tools.intel_lab import points


def test_front_laser_direction_and_range_cutoff():
    ranges = np.full(180, 81.83)
    ranges[[0, 30, 45, 80, 90, 179]] = [2, np.nan, 20, -1, 3, 4]
    expected = [[0, -2, 0], [3, 0, 0], [4*np.sin(np.deg2rad(1)), 4*np.cos(np.deg2rad(1)), 0]]
    np.testing.assert_allclose(points(ranges), expected, atol=1e-6)
