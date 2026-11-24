import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mgts_nav.vision import estimate


def scene(yaw=12, pitch=0, outliers=0, offset=0, narrow=False):
    rng = np.random.default_rng(921)
    mount = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    intrinsics = np.array([[550, 0, 320], [0, 550, 240], [0, 0, 1]])
    xy = rng.uniform([-.52, -.31], [.52, .31], (60, 2))
    if narrow:
        xy[:, 1] = 0
    depth = rng.uniform(2, 8, 60)
    points = np.column_stack((xy * depth[:, None], depth))
    moved = Rotation.from_euler('zy', [yaw, pitch], degrees=True).as_matrix()
    camera_displacement = np.array([offset, 0., 0.])
    next_points = (points - camera_displacement) @ (mount.T @ moved @ mount)
    current = points @ intrinsics.T
    later = next_points @ intrinsics.T
    pixel0 = current[:, :2] / current[:, 2:]
    pixel1 = later[:, :2] / later[:, 2:] + rng.normal(0, .08, (60, 2))
    pixel1[:outliers] += [105, -87]
    return dict(K=intrinsics.tolist(), base_from_camera_xyzw=Rotation.from_matrix(mount).as_quat().tolist(),
                tracks=[[i, *start.tolist(), *end.tolist()] for i, (start, end) in enumerate(zip(pixel0, pixel1))],
                camera_motion_bound_m=abs(offset), min_scene_range_m=2.)


@pytest.mark.parametrize('angle', [-12, 0, 12])
def test_base_yaw_direction_with_pixel_noise_and_outliers(angle):
    result = estimate(scene(yaw=angle, outliers=8, offset=.001))
    assert result['accepted']
    assert result['inliers'] == 52
    assert np.degrees(result['yaw_rad']) == pytest.approx(angle, abs=.08)


@pytest.mark.parametrize(('kwargs', 'why'), [
    ({'outliers': 18}, 'weak_rotation_consensus'),
    ({'offset': .15}, 'possible_parallax'),
    ({'pitch': 9}, 'nonplanar_rotation'),
    ({'narrow': True}, 'insufficient_angular_spread'),
])
def test_geometric_rejections(kwargs, why):
    result = estimate(scene(**kwargs))
    assert result['reason'] == why
    assert not result['accepted'] and result['yaw_rad'] is None


def test_parallax_invalidates_the_fit_even_if_external_bound_is_wrong():
    pair = scene(yaw=0, offset=.9)
    pair['camera_motion_bound_m'] = .001
    assert estimate(pair)['reason'] == 'weak_rotation_consensus'


def test_missing_external_bound_and_too_few_tracks():
    pair = scene()
    del pair['min_scene_range_m']
    assert estimate(pair)['reason'] == 'missing_motion_or_range_bound'
    pair['tracks'] = pair['tracks'][:4]
    assert estimate(pair)['reason'] == 'too_few_tracks'


def test_duplicate_ids_rejected():
    pair = scene()
    pair['tracks'].append(pair['tracks'][0])
    with pytest.raises(ValueError, match='unique'):
        estimate(pair)


def test_cli_runs_from_json(tmp_path):
    path = tmp_path / 'pair.json'
    path.write_text(json.dumps(scene(yaw=4)), encoding='utf-8')
    run = subprocess.run([sys.executable, '-m', 'mgts_nav.vision', str(path)],
                         check=True, capture_output=True, text=True)
    result = json.loads(run.stdout)
    assert result['accepted']
    assert np.degrees(result['yaw_rad']) == pytest.approx(4, abs=.05)
