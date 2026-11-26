import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mgts_nav.demo import PROFILE, generate
from mgts_nav.metrics import read_csv
from mgts_nav.solve import solve


def test_search_refusal_preserves_slot_diagnostics(monkeypatch, tmp_path):
    from mgts_nav import __main__ as cli
    from mgts_nav import delayed_start

    slots = [dict(index=0, score=.45, rejection='Weak final support')]
    def refuse(stream, method, pixels, meta, diagnostics):
        diagnostics['anchors'] = slots
        raise ValueError('Fewer than three accepted global poses')

    monkeypatch.setattr(cli, 'read_map', lambda _: (None, {}, []))
    monkeypatch.setattr(delayed_start, 'search_prefix', refuse)
    args = SimpleNamespace(command='localize', bag='unused', map='unused', config=PROFILE,
                           initial=None, initial_file=None, recover=False, confirm_start=None,
                           start_search='delayed')
    with pytest.raises(ValueError, match='^Start search refused: ') as error:
        cli.run(args, tmp_path)
    report = json.loads(str(error.value).removeprefix('Start search refused: '))
    assert report == dict(method='delayed', reason='Fewer than three accepted global poses', anchors=slots)


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


def test_start_search_conflict_is_caught_before_opening_bags():
    result = subprocess.run(
        [sys.executable, '-m', 'mgts_nav', 'localize', 'bag', '--map', 'map', '--config', 'cfg',
         '--out', 'unused', '--start-search', 'coarse', '--initial-file', 'pose'],
        capture_output=True, text=True)
    assert result.returncode == 2 and 'not allowed with argument' in result.stderr


def test_confirm_conflicts_with_start_search_before_bags(tmp_path):
    with pytest.raises(ValueError, match='automatic initialization'):
        solve('unused', 'unused', PROFILE, tmp_path / 'unused',
              start_search='coarse', confirm_start=True)
