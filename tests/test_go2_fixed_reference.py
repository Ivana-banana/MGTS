import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools import evaluate_go2_localization as evaluator


def test_fixed_reference_uses_odometry_and_compares_common_pairs(monkeypatch, tmp_path):
    times = (0., 20., 40.)
    odometry = [evaluator.Pose(Rotation.identity(), np.array([x, 0., 0.])) for x in (0., 1., 2.)]
    cloud = np.ones((3, 3))
    monkeypatch.setattr(evaluator, 'scans', lambda _: zip(times, [cloud] * 3, odometry, times))
    monkeypatch.setattr(evaluator, 'ANCHOR_STEP', 1)

    def reference_pose(_target, _source, guesses):
        assert set(guesses) == {'odometry', *(f'perturbed{i}' for i in range(6))}
        return np.array([1., 0., 0.]), ['odometry', 'perturbed0'], 1.

    monkeypatch.setattr(evaluator, 'reference_pose', reference_pose)
    full = tmp_path / 'full.csv'
    full.write_text('t,x,y,yaw\n0,0,0,0\n20,1,0,0\n40,0,0,0\n')
    partial = tmp_path / 'partial.csv'
    partial.write_text('t,x,y,yaw\n0,0,0,0\n20,1,0,0\n')

    baseline, before = evaluator.evaluate(tmp_path, {}, fixed_reference=True)
    report, after = evaluator.evaluate(tmp_path, {'full': full, 'partial': partial}, fixed_reference=True)

    assert [(r['i'], r['j'], r['reference']) for r in before] == [(0, 1, [1., 0., 0.]), (1, 2, [1., 0., 0.])]
    assert [{k: row[k] for k in base} for row, base in zip(after, before, strict=True)] == before
    assert report['revisits']['methods']['odometry'] == baseline['revisits']['methods']['odometry']
    assert report['revisits']['methods']['full']['over_threshold'] == 1
    common = report['revisits']['common']
    assert common['pairs'] == 1
    assert {name: stats['over_threshold'] for name, stats in common['methods'].items()} == {
        'odometry': 0, 'full': 0, 'partial': 0,
    }


def test_track_cannot_replace_reference_odometry(tmp_path):
    with pytest.raises(ValueError, match='reserved track name: odometry'):
        evaluator.evaluate(tmp_path, {'odometry': tmp_path / 'custom.csv'}, fixed_reference=True)
