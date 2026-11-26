import numpy as np
from scipy.spatial.transform import Rotation

from mgts_nav.localize3d import Pose
from tools.evaluate_go2_localization import longest_unmatched, on_scans, reference_pose, relative


def test_longest_unmatched_counts_seconds_and_start():
    times = np.arange(10) * 0.1
    labels = np.array(['matched', 'prediction_only', 'prediction_only', 'matched',
                       'prediction_only', 'prediction_only', 'prediction_only', 'matched',
                       'matched', 'prediction_only'])
    gap = longest_unmatched(times, labels, 0.1)
    assert gap == {'scans': 3, 'seconds': 0.3, 'starts_s': 0.4}


def test_trailing_gap_and_missing_rows():
    times = np.array([10.0, 10.1, 10.2, 10.3])
    track = (np.array([10.0, 10.1]), np.array([[0, 0, 0], [1, 0, 0]], float), np.array(['matched'] * 2))
    poses, labels = on_scans(times, track)
    assert list(labels) == ['matched', 'matched', 'missing', 'missing']
    assert np.isnan(poses[2:]).all()
    assert longest_unmatched(times, labels, 0.1) == {'scans': 2, 'seconds': 0.2, 'starts_s': 0.2}


def test_relative_is_frame_independent():
    a, b = np.array([1.0, 2.0, 0.3]), np.array([2.0, 2.5, 1.1])
    shift = np.array([5.0, -3.0, 0.7])

    def move(p):
        c, s = np.cos(shift[2]), np.sin(shift[2])
        return np.array([shift[0] + c * p[0] - s * p[1], shift[1] + s * p[0] + c * p[1], p[2] + shift[2]])

    assert np.allclose(relative(a, b), relative(move(a), move(b)))


def room(seed=3):
    rng = np.random.default_rng(seed)
    walls = []
    for x in np.arange(-3, 3, 0.04):
        for z in np.arange(0, 1.2, 0.08):
            walls += [[x, -2, z], [x, 2, z]]
    for y in np.arange(-2, 2, 0.04):
        for z in np.arange(0, 1.2, 0.08):
            walls += [[-3, y, z], [3, y, z]]
    box = rng.uniform([0.5, -1.0, 0.0], [1.0, -0.4, 0.6], (800, 3))
    return np.vstack([walls, box, rng.uniform([-3, -2, 0], [3, 2, 0], (2000, 3))])


def test_reference_recovers_known_motion_and_needs_distinct_guesses():
    target = room()
    truth = Pose(Rotation.from_euler('Z', 0.12), np.array([0.25, -0.15, 0.0]))
    source = truth.rotation.inv().apply(target - truth.xyz)
    near = Pose(Rotation.from_euler('Z', 0.10), np.array([0.20, -0.10, 0.0]))
    far = Pose(Rotation.from_euler('Z', 0.20), np.array([0.45, -0.05, 0.0]))
    pose, agreed, inliers = reference_pose(target, source, {'a': near, 'b': far})
    assert pose is not None and set(agreed) == {'a', 'b'} and inliers > 0.9
    assert np.allclose(pose, [0.25, -0.15, 0.12], atol=0.01)
    # One guess, or two near-identical guesses, cannot confirm a reference.
    twin = Pose(near.rotation, near.xyz + 0.01)
    assert reference_pose(target, source, {'a': near, 'c': twin})[0] is None
