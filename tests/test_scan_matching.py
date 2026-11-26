import numpy as np
from scipy.spatial import cKDTree

from mgts_nav.bag import Scan
from mgts_nav.slam import PointToLine, point_to_line_rows, rotation, track


def test_point_to_line_jacobian_matches_finite_differences():
    points = np.array([[1., 2.], [-2., 1.]])
    centers = np.array([[1., 0.], [-1., 2.]])
    normals = np.array([[.6, .8], [-.8, .6]])
    pose = np.array([.4, -.6, .3])
    lever = 2.5

    def rows(p):
        return point_to_line_rows(points @ rotation(p[2]).T, p[:2], centers, normals, lever)

    steps = np.diag([1., 1., 1. / lever]) * 1e-6
    numeric = np.column_stack([(rows(pose + d)[0] - rows(pose - d)[0]) / 2e-6 for d in steps])
    np.testing.assert_allclose(rows(pose)[1], numeric, atol=1e-9, rtol=0)


def test_point_to_line_keeps_along_corridor_prediction():
    x = np.arange(-2., 2.01, .08)
    wall = np.column_stack((x, np.ones_like(x)))
    points = np.vstack((wall, wall - [0., 2.]))
    scan = points[abs(points[:, 0]) < 1.]
    seed = np.array([.05, .10, 0.])

    pose, _ = PointToLine(cKDTree(points)).refine(scan, seed, seed)

    np.testing.assert_allclose(pose[0], seed[0], atol=1e-12, rtol=0)
    assert abs(pose[1]) < .015


def test_lidar_only_motion_does_not_use_previous_pose_as_motion_prior():
    x = np.arange(-2., 2.01, .08)
    wall = np.column_stack((x, np.full_like(x, 2.)))
    points = np.vstack((wall, wall[:, ::-1]))
    origin = np.zeros(2)
    motion = np.array([.095, -.11, .04])
    scans = [
        Scan(0., points, origin, None, None),
        Scan(.1, (points - motion[:2]) @ rotation(motion[2]), origin, None, None),
    ]

    first, moved = track(scans, np.zeros(3), map_points=points)

    assert first.status == moved.status == 'matched'
    np.testing.assert_allclose(moved.pose, motion, atol=.012, rtol=0)


def test_map_limits_translation_error_from_odometry():
    # With measured odometry the tracker keeps point-to-point ICP (Go2-safe path).
    # A 7 cm odom lie against a fixed scan is trimmed by ICP, not by point-to-line.
    x = np.arange(-2., 2.01, .08)
    wall = np.column_stack((x, np.full_like(x, 2.)))
    points = np.vstack((wall, wall[:, ::-1]))
    origin = np.zeros(2)
    scans = [
        Scan(0., points, origin, np.zeros(3), None),
        Scan(.1, points, origin, np.array([.07, 0., 0.]), 0.),
    ]

    _, result = track(scans, np.zeros(3), map_points=points)

    assert result.status == 'matched'
    assert abs(result.pose[0]) < .04


def test_odometry_coasts_through_lidar_outage_without_runaway():
    # With measured odometry, empty scans follow kinematics; no PointToLine bias path.
    x = np.arange(-2., 2.01, .08)
    wall = np.column_stack((x, np.full_like(x, 2.)))
    points = np.vstack((wall, wall[:, ::-1]))
    origin = np.zeros(2)
    empty = np.empty((0, 2))
    scans = [
        Scan(i / 10, empty if 20 <= i < 35 else points, origin, np.array([.03 * i, 0., 0.]), 0.)
        for i in range(36)
    ]

    rows = list(track(scans, np.zeros(3), map_points=points))

    assert rows[34].status == 'prediction_only'
    # Relative odom coasts ~0.03 m/step from a map-pinned pose near the origin.
    assert abs(rows[34].pose[0] - 0.45) < .05
    assert rows[35].status == 'matched'
    # Recovered match must not jump to the absolute odom reading (~1.05 m).
    assert abs(rows[-1].pose[0]) < .60


def test_accurate_odometry_along_corridor_does_not_run_away():
    # Go2-style corridor: walls constrain the cross-track axis; good odometry
    # along the corridor must stay near the kinematic pose (docs/go2-revisit-check.md).
    x = np.arange(-1., 12.01, .08)
    wall = np.column_stack((x, np.ones_like(x)))
    points = np.vstack((wall, wall - [0., 2.]))
    origin = np.zeros(2)
    scans = []
    for i in range(80):
        pose = np.array([.12 * i, 0., 0.])
        body = (points - pose[:2]) @ rotation(pose[2])
        visible = body[np.linalg.norm(body, axis=1) < 3.5]
        scans.append(Scan(i * .1, visible, origin, pose, 0.))

    rows = list(track(scans, np.zeros(3), map_points=points))

    assert sum(row.status == 'matched' for row in rows) >= 75
    np.testing.assert_allclose(rows[-1].pose[:2], scans[-1].odom[:2], atol=.25, rtol=0)
    assert abs(rows[-1].pose[0] - scans[-1].odom[0]) < 1.0
