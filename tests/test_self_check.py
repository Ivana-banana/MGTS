import json
import subprocess
import sys

import numpy as np
import pytest

from mgts_nav import self_check as sc
from mgts_nav.demo import PROFILE, generate


def test_cli_checks_profile_without_rewriting_it(tmp_path):
    data = tmp_path / 'office'
    generate(data)
    profile = tmp_path / 'profile.json'
    original = PROFILE.read_bytes()
    profile.write_bytes(original)
    command = [sys.executable, '-m', 'mgts_nav.self_check', data/'mapping', '--config', profile,
               '--max-scans', '80', '--budget', '180', '-q']
    report = tmp_path / 'report.json'
    subprocess.run([*command, '--out', report], check=True)
    observed = json.loads(report.read_bytes())
    assert observed['verdict'] == 'ok' and observed['complete']
    assert {'tilt', 'mirror', 'yaw', 'clock_odom', 'clock_imu'} == observed['checks'].keys()
    assert observed['limitations']
    assert subprocess.run([*command, '--out', profile]).returncode == 2
    sparse = tmp_path / 'sparse.json'
    assert subprocess.run([*command, '--max-scans', '1', '--out', sparse]).returncode == 3
    assert json.loads(sparse.read_bytes())['verdict'] == 'not_measured'
    assert profile.read_bytes() == original


@pytest.mark.parametrize('budget,count', [(0, 1), (float('nan'), 1), (1, 0)])
def test_invalid_limits_before_reading(budget, count):
    with pytest.raises(ValueError, match='positive and finite'):
        sc.self_check('unused', {}, budget_s=budget, max_scans=count)


def test_lag_has_sign_and_requires_excitation():
    t = np.arange(0., 12., .02)
    rate = np.sin(t**1.4) + .3*np.cos(t*2.7)
    lags, scores = sc.correlate(t, rate, t+.25, rate)
    assert lags[np.nanargmax(scores)] == pytest.approx(.25)
    _, scores = sc.correlate(t, rate, t, np.zeros(len(t)))
    assert not np.isfinite(scores).any()
