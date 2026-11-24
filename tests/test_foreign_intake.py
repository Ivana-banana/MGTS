"""Intake on recordings with unfamiliar conventions (tools/foreign_bags.py)."""
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mgts_nav.bag import inspect_bag, scans
from mgts_nav.calibration import extrinsics, fit_floor, floor_check, read_edges
from mgts_nav.profile import draft_profile
from tools.clock_lag import estimate
from tools.foreign_bags import SCENARIOS
from tools.split_recording import cut, stops


@pytest.fixture(scope='module')
def foreign(tmp_path_factory):
    root = tmp_path_factory.mktemp('foreign')
    for key, make in SCENARIOS.items():
        make(root / key, 23)
    return root


def expected(foreign, key):
    return json.loads((foreign / key / 'truth/expected_profile.json').read_text())


def same_rotation(a, b):
    return Rotation.from_quat(a).inv() * Rotation.from_quat(b)


@pytest.mark.parametrize('bag', ['mapping', 'inspection_run2.mcap'])
def test_chained_static_tf_and_bare_mcap_give_the_profile(foreign, bag):
    truth = expected(foreign, 'a')
    draft = draft_profile(foreign / 'a/data' / bag, verify_frames=True)['profile']
    for key in ('lidar', 'imu', 'odom', 'lidar_frame', 'imu_frame', 'base_frame', 'odom_frame'):
        assert draft[key] == truth[key]
    assert draft['base_from_lidar_xyz_m'] == pytest.approx(truth['base_from_lidar_xyz_m'])
    for key in ('base_from_lidar_xyzw', 'base_from_imu_xyzw'):
        assert same_rotation(draft[key], truth[key]).magnitude() < 1e-9


def test_combined_recording_is_cut_at_stops(foreign, tmp_path):
    session = foreign / 'b/data/session'
    found = stops(session, '/demo/odom')
    np.testing.assert_allclose(found, [[500, 501], [519, 524.05], [527.95, 531], [539, 540]], atol=0.06)
    counts = cut(session, tmp_path / 'inspection', 529.0, 540.0)
    assert counts == {'/demo/imu': 551, '/demo/odom': 221, '/demo/cloud': 110}
    lidar = next(r for r in inspect_bag(tmp_path / 'inspection') if r['topic'] == '/demo/cloud')
    assert 529 <= lidar['header_start_s'] and lidar['header_end_s'] <= 540


def test_float64_cloud_with_point_time_and_tilted_mount(foreign, tmp_path):
    data = foreign / 'c/data'
    lidar = next(r for r in inspect_bag(data / 'mapping') if r['topic'] == '/velodyne_points')
    assert lidar['field_types'] == dict(x='float64', y='float64', z='float64', intensity='float32',
                                        t='uint32', ring='uint16')
    assert lidar['cloud_layout']['height'] == 16 and not lidar['cloud_layout']['is_dense']
    draft = draft_profile(data / 'mapping')
    assert any("time field(s) ['t']" in q for q in draft['questions'])
    profile = dict(expected(foreign, 'c'))
    good = tmp_path / 'good.json'
    good.write_text(json.dumps(profile))
    level = floor_check(data / 'mapping', good)
    assert level['floor_observed'] and level['floor_tilt_deg'] < 0.5 and abs(level['floor_height_m']) < 0.03
    flat = tmp_path / 'untilted.json'
    flat.write_text(json.dumps(dict(profile, base_from_lidar_xyzw=[0, 0, 0, 1])))
    assert floor_check(data / 'mapping', flat)['floor_tilt_deg'] > 3
    micro = tmp_path / 'micro.json'
    micro.write_text(json.dumps(dict(profile, point_time=dict(field='t', unit='us'))))
    from mgts_nav.bag import load_config
    with pytest.raises(ValueError, match='point_time unit'):
        next(scans(data / 'mapping', load_config(micro)))


def test_robot_without_imu(foreign, tmp_path):
    data = foreign / 'd/data/mapping'
    truth = expected(foreign, 'd')
    draft = draft_profile(data)
    assert draft['profile']['imu'] is None and draft['profile']['base_from_imu_xyzw'] is None
    result = extrinsics(read_edges(data, '/tf_static'), 'base_link', 'laser')
    assert 'base_from_imu_xyzw' not in result
    assert same_rotation(result['base_from_lidar_xyzw'], truth['base_from_lidar_xyzw']).magnitude() < 1e-9
    profile = tmp_path / 'd.json'
    profile.write_text(json.dumps(truth))
    assert floor_check(data, profile)['floor_observed'] is False  # one horizontal ring sees no floor


def test_fit_floor_reports_signed_tilt():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-3, 3, (2000, 2))
    floor = np.column_stack([xy, np.tan(np.radians(4)) * xy[:, 0] + 0.02])
    wall = np.column_stack([np.full(500, 3.0), rng.uniform(-3, 3, 500), rng.uniform(0, 2, 500)])
    result = fit_floor(np.vstack([floor, wall]))
    assert result['floor_rise_pitch_deg'] == pytest.approx(4, abs=0.05)
    assert result['floor_height_m'] == pytest.approx(0.02, abs=1e-3)


def test_clock_lag_from_turns():
    t = np.arange(0, 30, 0.1)
    rate = np.where((t % 10) < 5, 0.4, -0.2)
    odom_t = np.arange(0, 30, 0.04)
    lagged = np.interp(odom_t - 0.25, t, rate)  # odometry headers run 0.25 s ahead
    lag, score = estimate(t, rate, odom_t, lagged)
    assert lag == pytest.approx(0.25, abs=0.02) and score > 0.9
