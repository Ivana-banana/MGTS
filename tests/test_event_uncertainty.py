import numpy as np

from mgts_nav.bag import Scan
from mgts_nav.event_uncertainty import assess


def test_crossing_boundary_and_missing_pose_are_distinct():
    scans = [Scan(float(t), np.array([[.1,.1]]), np.zeros(2), None, None) for t in (0,1)]
    report = assess(scans, {0.:np.array([.9,.5,0.])}, np.full((20,20),254),
                    dict(origin=[0,0,0],resolution=.1,free_thresh=.2),
                    {'office':[[0,0],[1,0],[1,1],[0,1]]}, [[.2,0,0]])
    assert [row['zones']['office'] for row in report['zones']] == ['uncertain','unobserved']
    assert report['unobserved_scans'] == [1.]
    assert report['obstacle_status'] == 'absent'
    assert report['obstacles'] == []
