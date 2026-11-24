"""Run both global searches on identical independent corridor scans and maps."""
from dataclasses import replace
import json
from unittest.mock import patch
from time import perf_counter

import numpy as np
import pytest

from mgts_nav.bag import Scan, load_config, scans
from mgts_nav.demo import PROFILE, generate
from mgts_nav.grid import Grid, read_map
from mgts_nav.initialization import estimate_initial
from mgts_nav.metrics import read_csv, wrap
from mgts_nav.slam import match, rotation, track


@pytest.fixture(scope='module', params=[41, 73])
def corridor(request, tmp_path_factory):
    seed = request.param
    directory = tmp_path_factory.mktemp(f'coarse-corridor-{seed}')
    data = directory / 'data'
    generate(data, layout='corridor', seed=seed)
    cfg = load_config(PROFILE)
    grid = Grid(.08)
    list(track(scans(data/'mapping', cfg), np.zeros(3), grid=grid))
    grid.save(directory)
    pixels, meta, _ = read_map(directory/'map.yaml')
    stream = list(scans(data/'inspection', cfg))
    truth = read_csv(data/'inspection_truth.csv')
    return seed, pixels, meta, stream, truth


def classify(pose, reference):
    xy = float(np.linalg.norm(pose[:2] - reference[:2]))
    angle = float(np.degrees(abs(wrap(pose[2] - reference[2]))))
    return dict(result='correct' if xy <= .25 and angle <= 10 else 'false',
                error_m=xy, error_deg=angle)


def measure(method, scan, pixels, meta, reference):
    stats = {}
    if method == 'coarse':
        from mgts_nav.coarse_start import estimate
    start = perf_counter()
    try:
        if method == 'baseline':
            with patch('mgts_nav.initialization.match', wraps=match) as counted:
                try:
                    pose = estimate_initial(scan.xy, pixels, meta)
                finally:
                    stats['icp'] = counted.call_count
        else:
            pose = estimate(scan, pixels, meta, stats)
    except ValueError as exc:
        elapsed = perf_counter() - start
        result = dict(result='refused', reason=str(exc))
    else:
        elapsed = perf_counter() - start
        assert np.shape(pose) == (3,) and np.all(np.isfinite(pose))
        result = classify(pose, reference)
    return dict(method=method, points=len(scan.xy), elapsed_s=elapsed, stats=stats, **result)


@pytest.mark.parametrize('method', ['baseline', 'coarse'])
def test_corridor_windows(corridor, method):
    seed, pixels, meta, stream, truth = corridor
    for view in ('full', '3m'):
        for index in (0, 30, 60):
            scan = stream[index]
            assert abs(scan.t - truth[index].t) < 1e-8
            if view == '3m':
                valid = np.linalg.norm(scan.xy - scan.origin, axis=1) <= 3.
                scan = replace(scan, xy=scan.xy[valid])
            # Both methods have a single cloud. The reference is used only to score outputs.
            scan = replace(scan, odom=None, gyro_delta=None)
            result = measure(method, scan, pixels, meta, np.array(truth[index][1:]))
            print(json.dumps(dict(seed=seed, view=view, index=index, **result)))
            if view == 'full':
                assert result['result'] == 'correct'
            if method == 'coarse' and view == '3m':
                assert result['result'] != 'false'
            if method == 'baseline' and (seed, view, index) == (73, '3m', 30):
                assert result['result'] == 'false' and result['error_m'] > 5


@pytest.mark.parametrize('method', ['baseline', 'coarse'])
def test_symmetric_corridor(method):
    # A rectangle invariant under a half-turn about its center; first returns only.
    pixels = np.full((31, 151), 254, dtype=np.uint8)
    pixels[[0, -1], :] = 0
    pixels[:, [0, -1]] = 0
    cell = .1
    meta = dict(origin=[0., 0., 0.], resolution=cell, occupied_thresh=.65, free_thresh=.196)
    pose = np.array([4.25, 1.15, .29])
    angles = np.arange(240) * (2*np.pi / 240)
    body = np.column_stack((np.cos(angles), np.sin(angles)))
    rays = body @ rotation(pose[2]).T
    lower = [.05, .05]
    upper = [(pixels.shape[1]-.5)*cell, (pixels.shape[0]-.5)*cell]
    walls = np.where(rays > 0, upper, lower)
    distances = ((walls - pose[:2]) / rays).min(axis=1)
    scan = Scan(0., body * distances[:, None], np.zeros(2), None, None)
    result = measure(method, scan, pixels, meta, pose)
    print(json.dumps(dict(seed=None, view='symmetric', index=0, **result)))
    assert result['result'] == 'refused', 'Indistinguishable poses in a symmetric rectangle'
    assert 'ambiguous' in result['reason'].lower()
