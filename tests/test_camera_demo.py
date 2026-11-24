import json

import numpy as np
import pytest

from mgts_nav.bag import load_config, scans
from mgts_nav.demo import PROFILE, START_NS, generate, pose_at
from mgts_nav.vision import estimate


def test_office_camera_observations_and_sensor_outages(tmp_path):
    root = tmp_path / 'recording'
    generate(root, seed=41, camera_experiment=True)
    with (root / 'inspection-pairs.jsonl').open() as stream:
        pairs = [json.loads(line) for line in stream]
    stream = list(scans(root / 'inspection', load_config(PROFILE)))
    assert len(pairs) == 4 * (len(stream) - 1)
    assert [row['t0'] for row in pairs[::4]] == [scan.t for scan in stream[:-1]]
    assert all(a['t1'] == b['t0'] for a, b in zip(pairs, pairs[1:]))
    assert pairs[-1]['t1'] == stream[-1].t
    assert all('yaw_rad' not in row and 'accepted' not in row for row in pairs)

    epoch = START_NS['inspection'] / 1e9
    weak = [scan for scan in stream if 4 <= scan.t - epoch < 6]
    assert len(weak) == 20 and all(len(scan.xy) <= 8 for scan in weak)
    assert any(scan.gyro_delta is None for scan in weak)
    assert any(scan.gyro_delta is not None for scan in weak)
    assert all(scan.odom is not None for scan in weak)
    row = pairs[175]
    measured = estimate(row)
    expected = pose_at('inspection', row['t1'] - epoch)[2] - pose_at('inspection', row['t0'] - epoch)[2]
    assert measured['accepted'] and measured['inliers'] < measured['tracks']
    assert measured['yaw_rad'] == pytest.approx(expected, abs=np.radians(.09))
    assert estimate(pairs[207])['reason'] == 'insufficient_angular_spread'
