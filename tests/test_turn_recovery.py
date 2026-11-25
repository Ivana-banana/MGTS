import numpy as np
from scipy.spatial import cKDTree

from mgts_nav.bag import Scan
from mgts_nav.slam import match
from tools.heap1_turn_recovery import AngularRecovery


def test_angular_proposal_requires_next_scan_agreement():
    points = np.random.default_rng(42).uniform([-4., -2.], [4., 2.], (100, 2))
    start = np.zeros(3)
    helper = AngularRecovery(cKDTree(points), start)
    scan = lambda i: Scan(.1*i, points, np.zeros(2), None, None)
    for i in range(3):
        assert helper.update(scan(i), False) is None
    assert helper.pending is not None
    np.testing.assert_allclose(helper.update(scan(3), False), start, atol=1e-8)
    assert helper.confirmations == [scan(3).t]


def test_periodic_corridor_does_not_confirm_wrong_bay():
    # Identical bays every metre. A four-metre error places the robot in a different bay.
    x = np.arange(-12., 12., .05)
    wall = 1.3 + .16*np.cos(2*np.pi*x)
    points = np.vstack([np.column_stack((x, sign*wall)) for sign in (-1, 1)])
    helper = AngularRecovery(cKDTree(points), np.array([4., 0., 0.]))
    for i in range(8):
        body = points[np.abs(points[:, 0]-.05*i) < 3.] - [.05*i, 0.]
        helper.last[0] = 4.+.05*i
        assert match(body, helper.tree, helper.last)[1] > .99
        assert helper.update(Scan(.1*i, body, np.zeros(2), None, None), False) is None
    assert helper.pending is None and not helper.confirmations
