import numpy as np
from scipy.spatial import cKDTree

from tools.intel_longitudinal import alignment


def test_current_scan_is_transformed_into_first_frame():
    first = np.array([[0., 1.], [1., 1.], [-1., 4.], [2., 3.]])
    # Sensor at (1, 2) facing +Y: local X = world Y - 2, local Y = 1 - world X.
    current = np.column_stack((first[:, 1] - 2, 1 - first[:, 0]))
    tree = cKDTree(first)
    score, mutual = alignment(first, tree, current, [1., 2., np.pi / 2])
    assert score < 1e-12 and mutual == len(first)
    assert alignment(first, tree, current, [-1., -2., -np.pi / 2])[0] > .1
