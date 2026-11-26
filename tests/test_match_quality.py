import numpy as np
from scipy.spatial.transform import Rotation

from mgts_nav.localize3d import Pose, SurfaceMap
from mgts_nav.match_quality import assess, eigen_ratio


def walls(xs, ys, z=np.arange(0.0, 1.2, 0.1)):
    return np.array([[x, y, h] for x in xs for y in ys for h in z])


def corridor():
    """Two long parallel walls along y and a floor: y is not observable."""
    along = np.arange(-6, 6, 0.1)
    sides = np.vstack([walls([-1.0], along), walls([1.0], along)])
    floor = np.array([[x, y, 0.0] for x in np.arange(-0.9, 0.95, 0.1) for y in along])
    return np.vstack([sides, floor])


def room():
    edge = np.arange(-2, 2.01, 0.1)
    floor = np.array([[x, y, 0.0] for x in edge for y in edge])
    return np.vstack([walls([-2.0, 2.0], edge), walls(edge, [-2.0, 2.0]), floor,
                      walls(np.arange(0.5, 1.0, 0.1), [0.5]), walls([0.5], np.arange(0.5, 1.0, 0.1))])


def test_corridor_is_degenerate_along_its_axis():
    points = corridor()
    surface = SurfaceMap(points)
    identity = Pose(Rotation.identity(), np.zeros(3))
    ratio, axis = eigen_ratio(surface, points[::3], identity)
    assert ratio < 0.008 and axis == 'y'
    q = assess(surface, points[::3], points[1::3], identity, identity, True)
    assert q.level == 'weak' and 'degenerate' in q.reasons


def test_closed_room_is_confident_and_large_jump_is_weak():
    points = room()
    surface = SurfaceMap(points)
    identity = Pose(Rotation.identity(), np.zeros(3))
    q = assess(surface, points[::3], points[1::3], identity, identity, True)
    assert q.level == 'confident', q.reasons
    jumped = Pose(Rotation.identity(), np.array([0.4, 0.0, 0.0]))
    q = assess(surface, points[::3], points[1::3], jumped, identity, True)
    assert 'large_correction' in q.reasons


def test_rejected_fit_is_prediction_only():
    surface = SurfaceMap(room())
    identity = Pose(Rotation.identity(), np.zeros(3))
    assert assess(surface, room(), room(), identity, identity, False).level == 'prediction_only'
