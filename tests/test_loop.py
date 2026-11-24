import numpy as np

from mgts_nav.loop import diagnose


def test_closed_lap_with_small_drift():
    xy = np.random.default_rng(42).normal(size=(120, 2))
    route = np.array([[0, 0, 0], [1.5, 0, 0], [.05, 0, 0]])
    saved = route.copy()
    result = diagnose(xy, xy, route)
    assert result['loop_supported'] and result['geometric_distance_m'] < .01
    assert result['endpoint_distance_m'] == .05
    np.testing.assert_array_equal(route, saved)


def test_open_route_with_identical_visible_landmarks():
    xy = np.random.default_rng(5).normal(size=(120, 2))
    result = diagnose(xy, xy - [2., 0.], [[0, 0, 0], [2, 0, 0]])
    assert not result['loop_supported'] and result['symmetric_overlap'] > .99
    assert abs(result['geometric_distance_m'] - 2.) < 1e-8


def test_static_and_missing_geometry():
    xy = np.zeros((2, 2))
    assert diagnose(xy, xy, np.zeros((2, 3)))['reason'] == 'no_excursion'
    assert diagnose(xy, xy, [[0, 0, 0], [2, 0, 0]])['reason'] == 'too_few_endpoints'
