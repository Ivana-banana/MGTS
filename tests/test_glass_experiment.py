import numpy as np

from mgts_nav.glass_experiment import PANEL, RADIUS, STOP, modify_hits, person_center


def test_person_occludes_missing_glass_only_if_in_front():
    origin = np.array([0., 0.])
    hits = np.array([[0., -2.], [1., -2.], [0., -.4], [2., -2.]])
    changed, keep = modify_hits(origin, hits, True, np.array([0., -1.]), np.zeros(4))
    np.testing.assert_array_equal(keep, [True, False, True, True])
    np.testing.assert_allclose(changed[0], [0, -1. + RADIUS])
    np.testing.assert_array_equal(changed[2:], hits[2:])


def test_control_and_pause():
    changed, keep = modify_hits(np.zeros(2), PANEL.copy(), False, None, np.zeros(2))
    np.testing.assert_array_equal(changed, PANEL)
    assert keep.all()
    np.testing.assert_array_equal(person_center(4), STOP)
    np.testing.assert_array_equal(person_center(16), STOP)
    assert np.linalg.norm(person_center(20) - STOP) > 2 * RADIUS
