import numpy as np
import pytest

from mgts_nav.map_eval import evaluate_map, segment_distances


def test_distance_includes_finite_endpoints():
    segment = np.array([[[0., 0.], [1., 0.]]])
    np.testing.assert_allclose(segment_distances(np.array([[0.5, 1.], [4., 4.]]), segment), [1, 5])


def test_false_occupancy_and_static_wall_coverage():
    segments = np.array([[[0., 0.], [1., 0.]], [[10., 0.], [13., 0.]]])
    points = np.array([[x / 10, 0] for x in range(11)] + [[1, -0.9], [4, 4]])
    result = evaluate_map(points, segments, 0.1, 0.12)
    assert result["off_wall_in_person_sweep"] == result["off_wall_outside_person_sweep"] == 1
    assert result["off_wall_fraction"] == pytest.approx(2 / 13)
    assert result["static_wall_length_covered_fraction"] == pytest.approx(0.25)
