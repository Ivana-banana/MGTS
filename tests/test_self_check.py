"""Self-check of a sensor profile without reference (mgts_nav.self_check)."""
import json
import subprocess
import sys

import pytest

from mgts_nav import self_check as sc
from mgts_nav.bag import load_config
from mgts_nav.demo import PROFILE, generate
from tools.foreign_bags import SCENARIOS


@pytest.fixture(scope='module')
def foreign(tmp_path_factory):
    root = tmp_path_factory.mktemp('self-check')
    for key in ('d', 'e'):
        SCENARIOS[key](root / key, 23)
    return root


def recording(foreign, key, max_scans=80):
    cfg = load_config(foreign / key / 'truth/expected_profile.json')
    memory, _ = sc.load_recording(foreign / key / 'data/mapping', cfg, max_scans)
    return cfg, memory


def score(memory, cfg):
    return sc.consistency(*sc.build(memory, cfg))['score']


def test_wrong_tilt_sign_breaks_map_consistency_of_a_flat_scan(foreign):
    cfg, memory = recording(foreign, 'e')
    good = score(memory, cfg)
    flipped = score(memory, sc.with_rotation(cfg, [0, -12, 0]))
    forgotten = score(memory, sc.with_rotation(cfg, [0, -6, 0]))
    assert good > 0.85
    assert good - flipped > 0.15 and good - forgotten > 0.05


def test_tilt_search_finds_sign_and_size(foreign, monkeypatch):
    cfg, memory = recording(foreign, 'e')
    monkeypatch.setattr(sc, 'COARSE_DEG', (-12, -6, 6, 12))
    claimed = sc.with_rotation(cfg, [0, -12, 0])  # -6 deg instead of +6 deg
    base = sc.consistency(*sc.build(memory, claimed))
    result, delta = sc.sweep_tilt(memory, claimed, base, sc.Budget(60), log=lambda m: None)
    assert result['status'] == 'warning'
    assert delta[0] == pytest.approx(0, abs=3) and delta[1] == pytest.approx(12, abs=3)


def test_upside_down_lidar_is_found(foreign, monkeypatch):
    monkeypatch.setattr(sc, 'COARSE_DEG', ())  # no tilt search: only the mirror hypothesis
    cfg = load_config(foreign / 'd/truth/expected_profile.json')
    report, fixed = sc.self_check(foreign / 'd/data/mapping', dict(cfg, base_from_lidar_xyzw=[0, 0, 0, 1]),
                                  max_scans=80, log=lambda m: None)
    assert report['warnings'] == ['mirror']
    assert (sc.Rotation.from_quat(fixed['base_from_lidar_xyzw']).inv()
            * sc.Rotation.from_quat(cfg['base_from_lidar_xyzw'])).magnitude() < 1e-6


def test_odometry_clock_offset(foreign):
    cfg, memory = recording(foreign, 'd')
    late = dict(cfg, header_offset_s={})
    _, estimates = sc.build(memory, late)
    clock = sc.rate_check(memory, late, estimates, 'odom')
    assert clock['status'] == 'warning' and clock['header_offset_s'] == pytest.approx(-0.25, abs=0.04)
    _, estimates = sc.build(memory, cfg)
    assert sc.rate_check(memory, cfg, estimates, 'odom')['status'] == 'ok'


def test_lidar_yaw_from_travel_direction(foreign):
    cfg, memory = recording(foreign, 'e')
    turned = sc.with_rotation(cfg, [0, 0, 6])
    _, estimates = sc.build(memory, turned)
    found = sc.heading_check(memory, turned, estimates)
    assert found['status'] == 'warning' and found['correction_yaw_deg'] == pytest.approx(-6, abs=2.5)
    _, estimates = sc.build(memory, cfg)
    assert sc.heading_check(memory, cfg, estimates)['status'] == 'ok'


def test_cli_confirms_a_correct_profile(foreign, tmp_path):
    report = tmp_path / 'report.json'
    done = subprocess.run([sys.executable, '-m', 'mgts_nav.self_check', foreign / 'e/data/mapping',
                           '--config', foreign / 'e/truth/expected_profile.json', '--out', report,
                           '--max-scans', '60', '-q'], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    result = json.loads(report.read_text())
    assert result['verdict'] == 'ok' and result['warnings'] == []
    assert {'mirror', 'tilt', 'yaw', 'clock_imu', 'clock_odom'} <= result['checks'].keys()
    assert 'verdict: no inconsistency found' in done.stdout


def test_map_reports_rejections_when_no_scan_is_usable(tmp_path):
    data = tmp_path / 'demo'
    generate(data, seed=17)
    profile = json.loads(PROFILE.read_text())
    config = tmp_path / 'high-band.json'
    config.write_text(json.dumps(dict(profile, height_band_m=[5.0, 6.0])))
    done = subprocess.run([sys.executable, '-m', 'mgts_nav', 'map', data / 'mapping', '--config', config,
                           '--out', tmp_path / 'map'], capture_output=True, text=True)
    assert done.returncode == 2
    assert 'No usable scan contributed to the map' in done.stderr
    assert '"outside_height_band": 200' in done.stderr and 'used: 0' in done.stderr
