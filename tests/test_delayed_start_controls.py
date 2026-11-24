"""Blind controls for the fixed 51-scan delayed-start window."""
from dataclasses import replace
import json
from time import perf_counter

import numpy as np
import pytest

from mgts_nav.bag import Scan, load_config, scans
from mgts_nav.demo import PROFILE, generate
from mgts_nav.grid import Grid, read_map
from mgts_nav.metrics import Pose, evaluate, read_csv, wrap
from mgts_nav.slam import relative, rotation, track


@pytest.fixture(scope="module")
def maps(tmp_path_factory):
    result = {}
    config = load_config(PROFILE)
    for layout, seed in (("office", 17), ("corridor", 41), ("corridor", 73)):
        root = tmp_path_factory.mktemp(f"{layout}-{seed}")
        data = root / "data"
        generate(data, layout=layout, seed=seed)
        grid = Grid(.08)
        list(track(scans(data / "mapping", config), np.zeros(3), grid=grid))
        grid.save(root)
        pixels, meta, points = read_map(root / "map.yaml")
        inspection_config = dict(config, odom=None)
        result[layout, seed] = (pixels, meta, list(scans(data / "inspection", inspection_config)),
                                data / "inspection_truth.csv", points)
    return result


@pytest.fixture(params=['pairwise', 'keyframes'])
def local_method(request):
    return request.param


def estimate(window, pixels, meta, local_method):
    from mgts_nav.delayed_start import estimate_start
    assert len(window) == 51 and all(scan.odom is None for scan in window)
    stats = {}
    start = perf_counter()
    try:
        pose = estimate_start(window, pixels, meta, stats, local_method=local_method)
    except ValueError as exc:
        pose, reason = None, str(exc)
    else:
        assert np.shape(pose) == (3,) and np.all(np.isfinite(pose))
        reason = None
    return pose, dict(seconds=round(perf_counter() - start, 3), reason=reason,
                      local_method=local_method, local_end=stats.get('local_end'))


@pytest.mark.parametrize("layout,seed,index,radius,foreign", [
    ("office", 17, 0, None, False),
    ("office", 17, 0, None, True),
    ("corridor", 41, 0, None, False),
    ("corridor", 41, 30, 3., False),
    ("corridor", 73, 0, None, False),
    ("corridor", 73, 30, 3., False),
])
def test_recorded_windows(maps, layout, seed, index, radius, foreign, local_method):
    pixels, meta, stream, truth_path, points = maps[layout, seed]
    if foreign:
        pixels, meta = maps["corridor", 41][:2]
    window = []
    for scan in stream[index:index+51]:
        xy = scan.xy
        if radius is not None:
            xy = xy[np.linalg.norm(xy - scan.origin, axis=1) <= radius]
        window.append(replace(scan, xy=xy))
    pose, report = estimate(window, pixels, meta, local_method)
    truth = read_csv(truth_path)
    expected_motion = relative(np.array(truth[index][1:]), np.array(truth[index+50][1:]))
    if report['local_end'] is not None:
        report['local_end_error_m'] = float(np.linalg.norm(expected_motion[:2] - report['local_end'][:2]))
    result = "refused"
    if pose is not None:
        if foreign:
            result = "false"
        else:
            ref = truth[index]
            assert abs(ref.t - window[0].t) < 1e-8
            xy = float(np.linalg.norm(pose[:2] - ref[1:3]))
            yaw = float(np.degrees(abs(wrap(pose[2] - ref.yaw))))
            report.update(xy_m=xy, yaw_deg=yaw)
            route = [Pose(row.t, *row.pose) for row in track(stream[index:], pose, map_points=points)]
            report['ate_m'] = evaluate(route, truth[index:])['ate_xy_rmse_m']
            result = "correct" if xy <= .25 and yaw <= 10 else "false"
    print(json.dumps(dict(layout=layout, seed=seed, index=index, radius=radius,
                          foreign=foreign, result=result, **report)))
    assert result != "false"
    if local_method == 'keyframes':
        assert result == ('refused' if foreign or radius is not None else 'correct')


def test_symmetric_moving_window(local_method):
    # Rectangular walls give identical ranges along the half-turned trajectory.
    pixels = np.full((31, 151), 254, dtype=np.uint8)
    pixels[[0, -1], :] = 0
    pixels[:, [0, -1]] = 0
    meta = dict(origin=[0., 0., 0.], resolution=.1, occupied_thresh=.65, free_thresh=.196)
    angles = np.arange(240) * (2*np.pi / 240)
    body = np.column_stack((np.cos(angles), np.sin(angles)))
    window = []
    for i in range(51):
        position = np.array([4.25 + .04*i, 1.15 + .005*i])
        rays = body @ rotation(.29).T
        walls = np.where(rays > 0, [15.05, 3.05], [.05, .05])
        distance = ((walls - position) / rays).min(axis=1)
        window.append(Scan(.1*i, body * distance[:, None], np.zeros(2), None, 0.))
    pose, report = estimate(window, pixels, meta, local_method)
    print(json.dumps(dict(layout="rectangle", result="refused" if pose is None else "false", **report)))
    assert pose is None
