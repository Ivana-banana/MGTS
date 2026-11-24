from dataclasses import replace

import numpy as np

from mgts_nav.bag import Scan
from mgts_nav.motion_disagreement import windows
from mgts_nav.slam import Estimate


def test_motion_channels_and_unobservable_common_bias():
    stream, poses = [], []
    for i in range(11):
        x = i / 10
        stream.append(Scan(x, np.empty((0, 2)), np.zeros(2), np.array([3., 2+x, np.pi/2]), 0.))
        poses.append(Estimate(x, np.array([5+x, 3., 0.]), 'matched', 1.))
    result = list(windows(stream, poses))
    assert all(row['alert'] is None for row in result[:-1])
    assert result[-1]['translation_m'] < 1e-12 and not result[-1]['alert']
    broken = [replace(s, odom=None, gyro_delta=.02) for s in stream]
    row = list(windows(broken, poses))[-1]
    assert row['alert'] and row['translation_m'] is None and 11 < row['yaw_deg'] < 12
    broken[5] = replace(broken[5], gyro_delta=None)
    assert list(windows(broken, poses))[-1]['alert'] is None
    stream[-1] = replace(stream[-1], odom=stream[-1].odom + [0., 1., 0.])
    assert list(windows(stream, poses))[-1]['alert']
