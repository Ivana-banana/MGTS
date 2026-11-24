import json
import subprocess
import sys

import numpy as np
import pytest

from mgts_nav.demo import PROFILE, generate
from mgts_nav.metrics import read_csv
from mgts_nav.solve import solve


def test_solve_replays_entire_search_window(tmp_path):
    data = tmp_path / 'bag'
    generate(data)
    truth = read_csv(data / 'inspection_truth.csv')
    (data / 'initial_pose.json').unlink()
    (data / 'inspection_truth.csv').unlink()
    (data / 'mapping_truth.csv').unlink()
    out = tmp_path / 'result'
    solve(data/'mapping', data/'inspection', PROFILE, out, start_search='delayed-keyframes')
    result = out/'inspection'
    summary = json.loads((result/'summary.json').read_text(encoding='utf-8'))
    assert summary['initialization'] == 'delayed-keyframes'
    assert len(summary['start_search']['anchors']) == 6
    assert [r.t for r in read_csv(result/'trajectory.csv')] == [r.t for r in truth]
    assert np.linalg.norm(np.array(summary['initial_pose'][:2])-truth[0][1:3]) < .25


@pytest.mark.parametrize('command, flags', [
    (['mgts_nav', 'localize', 'bag', '--map', 'map'], ['--initial-file', 'pose']),
    (['mgts_nav.solve', 'mapping', 'inspection'], ['--confirm-start']),
])
def test_start_search_conflict_is_caught_before_opening_bags(command, flags):
    result = subprocess.run([sys.executable, '-m', *command, '--config', 'cfg', '--out', 'unused',
                             '--start-search', 'coarse', *flags], capture_output=True, text=True)
    assert result.returncode == 2 and 'not allowed with argument' in result.stderr
