import json
import subprocess
import sys

import pytest

from mgts_nav.bag import inspect_bag
from mgts_nav.demo import PROFILE, generate
from mgts_nav.metrics import evaluate, read_csv
from mgts_nav.solve import solve


@pytest.mark.parametrize("confirm", [False, True])
def test_two_bags_without_any_ground_truth_files_or_start(tmp_path, confirm):
    data = tmp_path / 'recordings'
    generate(data)
    truth = read_csv(data / 'inspection_truth.csv')
    for name in ('inspection_truth.csv', 'mapping_truth.csv', 'initial_pose.json'):
        (data / name).unlink()
    result = tmp_path / 'solution'
    if confirm:
        subprocess.run([sys.executable, '-m', 'mgts_nav.solve', str(data / 'mapping'),
                        str(data / 'inspection'), '--config', str(PROFILE), '--out', str(result),
                        '--confirm-start'], check=True)
    else:
        solve(data / 'mapping', data / 'inspection', PROFILE, result)
    summary = json.loads((result / 'inspection' / 'summary.json').read_text())
    assert summary['initialization'] == ('confirmed_scan_to_map' if confirm else 'scan_to_map')
    inspection = read_csv(result / 'inspection' / 'trajectory.csv')
    assert [row.t for row in inspection] == [row.t for row in truth]
    assert summary['status']['matched'] >= 95
    metrics = evaluate(read_csv(result / 'inspection' / 'continuous.csv'), truth)
    assert metrics['coverage'] == 1 and metrics['ate_xy_rmse_m'] < .09
    assert {row['topic'] for row in inspect_bag(result / 'rviz', inventory=True)} == {'/map', '/pose', '/path'}
