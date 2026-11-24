import json
import subprocess
import sys
from time import perf_counter

import numpy as np
import pytest
from rosbags.typesys import Stores, get_typestore

from mgts_nav.bag import Scan
from mgts_nav.demo import PROFILE, record
from mgts_nav.interior_loop import search
from mgts_nav.metrics import read_csv


@pytest.mark.parametrize('returns', [True, False])
def test_office_route(tmp_path, returns):
    knots = [0, 1, 5, 9, 15, 19, 20]
    points = np.array([[0, 0], [0, 0], [2.4, 0], [2.4, 2], [0, 0], [-2, 1], [-2, 1]])

    def route(run, t, layout, duration):
        if returns:
            return np.array([np.interp(t, knots, values) for values in points.T] + [0.])
        return np.array([np.clip((t-1)/18, 0, 1)*4.3, 0., 0.])

    record(tmp_path, 'mapping', json.loads(PROFILE.read_text()), get_typestore(Stores.ROS2_HUMBLE),
           59, trajectory=route)
    bag = tmp_path / 'mapping'
    out = tmp_path / 'map'
    subprocess.run([sys.executable, '-m', 'mgts_nav', 'map', str(bag), '--config', str(PROFILE),
                    '--out', str(out)], check=True, capture_output=True)
    tick = perf_counter()
    payload = subprocess.check_output([sys.executable, '-m', 'mgts_nav.interior_loop', str(bag),
                                      str(out/'trajectory.csv'), '--config', str(PROFILE)])
    seconds = perf_counter() - tick
    report = json.loads(payload)
    truth = read_csv(tmp_path/'mapping_truth.csv')
    candidates = report['candidates']
    assert np.linalg.norm(np.array(truth[0][1:3])-truth[-1][1:3]) > 2
    for pair in candidates:
        a, b = [truth[pair[key]] for key in ('first', 'second')]
        assert np.linalg.norm(np.array(a[1:3])-b[1:3]) <= .5
    assert bool(candidates) == returns
    if returns:
        assert any(p['first'] <= 15 and 140 <= p['second'] <= 160 for p in candidates)
    print(returns, report['counts'], len(candidates), round(seconds, 3))


def test_periodic_corridor_and_budget():
    # Distinct physical locations 4 m apart share the same periodic views.
    # The pose prior aliases them after an excursion, so proximity alone would accept.
    x = np.arange(-8., 8., .05)
    teeth = .15 * np.cos(2*np.pi*x)
    xy = np.vstack([np.column_stack((x, side * (1.3 + teeth))) for side in (-1, 1)])
    positions = [0., 1., 2., 3., 2., 1., 0., 0.]
    scans = [Scan(float(i), xy, np.zeros(2), None, None) for i in range(len(positions))]
    rows = [dict(t=i, x=x, y=0, yaw=0, status='matched') for i, x in enumerate(positions)]
    full = search(scans, rows, stride=1)
    print('corridor full', full['counts'])
    assert not full['candidates'] and 'skipped_budget' not in full['counts']
    report = search(scans, rows, stride=1, budget=2)
    print('corridor', report['counts'])
    assert not report['candidates']
    assert report['counts']['ambiguous_translation'] > 0
    assert report['counts']['tested'] == 2 and report['counts']['skipped_budget'] > 0
    rows[0]['status'] = 'prediction_only'
    report = search(scans, rows, stride=1, history=3)
    assert report['counts']['skipped_unconfirmed_or_sparse'] == 1
    assert report['counts']['evicted'] == 4
    with pytest.raises(ValueError, match='lengths differ'):
        search(scans, rows[:-1])
