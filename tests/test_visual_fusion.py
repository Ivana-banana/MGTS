import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mgts_nav.bag import Scan
from mgts_nav.visual_fusion import apply_visual


def pair(t0, t1, yaw=4, **overrides):
    # Optical z coincides with base z on this upward-facing synthetic camera.
    rng = np.random.default_rng(23)
    xy = rng.uniform(-.45, .45, (25, 2))
    directions = np.column_stack((xy, np.ones(len(xy))))
    after = Rotation.from_euler('z', -yaw, degrees=True).apply(directions)
    tracks = [[i, *p, *q[:2]] for i, (p, q) in enumerate(zip(xy, after))]
    return dict(t0=t0, t1=t1, K=np.eye(3).tolist(), base_from_camera_xyzw=[0, 0, 0, 1],
                tracks=tracks, camera_motion_bound_m=0., min_scene_range_m=1., **overrides)


def run(tmp_path, pairs, times, gyro):
    path = tmp_path / 'pairs.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in pairs), encoding='utf-8')
    raw = [Scan(t, np.ones((1, 2)), np.zeros(2), None, g) for t, g in zip(times, gyro)]
    stats = {}
    return list(apply_visual(raw, path, stats)), stats, raw


def test_real_geometry_fallback_blend_and_imu_conflict(tmp_path):
    # A LiDAR interval may straddle two adjacent camera pairs. Both must be used.
    pairs = [pair(0., .1, 2), pair(.1, .2, 6), pair(.2, .3, 2), pair(.3, .4, 6)]
    scans, stats, raw = run(tmp_path, pairs, [.05, .15, .25, .35],
                            [None, None, np.radians(2), np.radians(-10)])
    assert scans[0] is raw[0] and raw[1].gyro_delta is None
    np.testing.assert_allclose(np.degrees([s.gyro_delta for s in scans[1:]]), [4., 2.4, -10.])
    assert scans[1].xy is raw[1].xy
    assert stats['fallback'] == stats['blend'] == stats['conflict'] == stats['rejected'] == 1


def test_gap_and_partial_outside_do_not_extrapolate(tmp_path):
    pairs = [pair(.05, .1), pair(.12, .2)]
    scans, stats, raw = run(tmp_path, pairs, [.04, .08, .15, .22], [None] * 4)
    assert all(s is r for s, r in zip(scans, raw))
    assert stats['uncovered'] == stats['rejected'] == 3


def test_long_and_geometry_rejections_do_not_supply_yaw(tmp_path):
    too_long = pair(0, .35)
    unsupported = pair(.35, .45)
    unsupported.update(tracks=[], accepted=True, yaw_rad=.75)
    output, stats, _ = run(tmp_path, [too_long, unsupported], [0, .35, .45], [None]*3)
    assert stats['long_pairs'] == stats['geometry_rejected_pairs'] == 1
    assert stats['accepted_pairs'] == 0
    assert stats['geometry_rejection_reasons'] == {'too_few_tracks': 1}
    assert stats['rejected'] == 2 and all(scan.gyro_delta is None for scan in output)


@pytest.mark.parametrize('bad_pairs', [
    [(0., .1), (.05, .15)], [(0., .1), (0., .1)], [(.25, .3), (0, .1)], [(1, 1)]])
def test_invalid_camera_clock_fails_closed(tmp_path, bad_pairs):
    with pytest.raises(ValueError, match=r'pairs.jsonl:\d+'):
        run(tmp_path, [pair(a, b) for a, b in bad_pairs], [0, .1], [None]*2)


def test_repeated_lidar_time_rejected(tmp_path):
    with pytest.raises(ValueError, match='strictly increasing'):
        run(tmp_path, [], [1, 1], [None, None])
