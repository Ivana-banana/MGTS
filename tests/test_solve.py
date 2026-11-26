import json
import subprocess
import sys

import pytest

from mgts_nav.bag import inspect_bag
from mgts_nav.demo import PROFILE, generate
from mgts_nav.metrics import evaluate, read_csv
from mgts_nav.solve import solve
from mgts_nav.slam import read_initial_pose


@pytest.mark.parametrize('content', ['[1,2]', '[0,1e999,0]', '[true,0,0]', '{"x":1}', 'broken'])
def test_invalid_pose_file_names_path(tmp_path, content):
    path = tmp_path / 'start.json'
    path.write_text(content, encoding='utf-8-sig')
    with pytest.raises(ValueError, match='start.json'):
        read_initial_pose(path)


@pytest.mark.parametrize('resolution', [0.009, 1.01, float('nan'), float('inf')])
def test_invalid_resolution_rejected_before_bags(tmp_path, resolution):
    with pytest.raises(ValueError, match='resolution'):
        solve('unused', 'unused', PROFILE, tmp_path/'result', resolution=resolution)
    assert not (tmp_path/'result').exists()


def test_nonfinite_start_rejected_before_bags(tmp_path):
    with pytest.raises(ValueError, match='finite x, y, yaw'):
        solve('unused', 'unused', PROFILE, tmp_path/'result', initial=[1, 2, float('nan')])
    assert not (tmp_path/'result').exists()


@pytest.mark.parametrize("confirm", [False, True])
def test_two_bags_without_any_ground_truth_files_or_start(tmp_path, confirm):
    data = tmp_path / 'recordings'
    generate(data)
    truth = read_csv(data / 'inspection_truth.csv')
    for name in ('inspection_truth.csv', 'mapping_truth.csv', 'initial_pose.json'):
        (data / name).unlink()
    result = tmp_path / 'solution'
    if confirm:
        # Default path: automatic start is confirmed, recovery stays on.
        subprocess.run([sys.executable, '-m', 'mgts_nav.solve', str(data / 'mapping'),
                        str(data / 'inspection'), '--config', str(PROFILE), '--out', str(result),
                        '--mapping-velocity-prior'], check=True)
    else:
        solve(data / 'mapping', data / 'inspection', PROFILE, result, confirm_start=False)
    summary = json.loads((result / 'inspection' / 'summary.json').read_text())
    map_summary = json.loads((result / 'map' / 'summary.json').read_text())
    options = json.loads((result / 'solve-options.json').read_text())
    if confirm:
        assert map_summary['mapping_velocity_prior_steps'] == 0
        assert options['confirm_start'] is True
        assert options['recover'] is True
    else:
        assert 'mapping_velocity_prior_steps' not in map_summary
        assert options['confirm_start'] is False
    assert 'mapping_velocity_prior_steps' not in summary
    assert summary['initialization'] == ('confirmed_scan_to_map' if confirm else 'scan_to_map')
    inspection = read_csv(result / 'inspection' / 'trajectory.csv')
    assert [row.t for row in inspection] == [row.t for row in truth]
    assert summary['status']['matched'] >= 95
    metrics = evaluate(read_csv(result / 'inspection' / 'continuous.csv'), truth)
    assert metrics['coverage'] == 1 and metrics['ate_xy_rmse_m'] < .09
    assert {row['topic'] for row in inspect_bag(result / 'rviz', inventory=True)} == {'/map', '/pose', '/path'}
