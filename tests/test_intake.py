import json

import pytest

from mgts_nav.bag import load_config
from mgts_nav.demo import PROFILE, generate
from mgts_nav.solve import solve


@pytest.mark.parametrize(('field', 'value'), [
    ('base_from_lidar_xyzw', [0, 0, 0, 0]), ('base_from_lidar_xyz_m', None),
    ('stationary_s', None), ('max_sync_gap_s', float('nan')), ('height_band_m', [0]),
    ('lidar_frame', ''), ('point_time', []), ('header_offset_s', {'imu': 'unknown'}),
])
def test_incomplete_or_invalid_profile_names_the_field(tmp_path, field, value):
    profile = json.loads(PROFILE.read_text())
    profile[field] = value
    path = tmp_path / 'inspection.json'
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match=field):
        load_config(path)


def test_bad_second_recording_rejected_before_map_and_can_retry(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    generate(data)
    profile = json.loads(PROFILE.read_text())
    profile['lidar'] = '/unavailable'
    path = tmp_path / 'inspection.json'
    path.write_text(json.dumps(profile))
    result = tmp_path / 'solution'
    monkeypatch.setattr('mgts_nav.solve.subprocess.run',
                        lambda *a, **k: pytest.fail('A navigation subprocess started before intake finished'))
    with pytest.raises(ValueError, match=r'inspection.*Expected .*PointCloud2 on /unavailable'):
        solve(data / 'mapping', data / 'inspection', PROFILE, result, inspection_profile=path)
    assert not result.exists()


def test_explicitly_disabled_sensors_do_not_require_their_calibration(tmp_path):
    profile = json.loads(PROFILE.read_text())
    profile.update(imu=None, odom=None)
    for key in ('base_from_imu_xyzw', 'imu_frame', 'odom_frame', 'base_frame'):
        del profile[key]
    path = tmp_path / 'lidar-only.json'
    path.write_text(json.dumps(profile))
    assert load_config(path) == profile
