import numpy as np
import pytest

from mgts_nav.bag import Scan
from mgts_nav.recovery import Recovery
from mgts_nav.slam import rotation, track


def scene(asymmetric=True):
    pixels = np.full((41, 61), 254, dtype=np.uint8)
    pixels[[0, -1], :] = 0
    pixels[:, [0, -1]] = 0
    if asymmetric:
        pixels[8:21, 12] = 0
        pixels[20, 12:28] = 0
    meta = dict(origin=[-2., -2., 0.], resolution=.1, occupied_thresh=.65, free_thresh=.196)
    walls = (np.argwhere(pixels[::-1] == 0)[:, ::-1] + .5) * .1 - 2
    poses = [np.array([.50 + .04 * i, -.27, .38]) for i in range(5)]
    scans = [Scan(i * .1, (walls - pose[:2]) @ rotation(pose[2]), np.zeros(2), pose, None)
             for i, pose in enumerate(poses)]
    return Recovery(pixels, meta), walls, poses, scans


@pytest.mark.parametrize('asymmetric', [True, False])
def test_track_after_loss(asymmetric):
    recovery, walls, poses, scans = scene(asymmetric)
    rows = list(track(scans, [50., 50., 0.], map_points=walls, recovery=recovery))
    assert recovery.stats == {'attempts': 1, 'recoveries': int(asymmetric)}
    assert [row.status for row in rows[:3]] == ['prediction_only'] * 3
    if asymmetric:
        assert [row.status for row in rows[3:]] == ['matched', 'matched']
        np.testing.assert_allclose(rows[3].pose, poses[3], atol=.05)
    else:
        assert all(row.status == 'prediction_only' for row in rows)


@pytest.mark.parametrize('interruption', ['matched', 'duplicate', 'gap', 'disagreement'])
def test_unconfirmed_candidate_is_discarded(interruption, monkeypatch):
    recovery, _, poses, scans = scene()
    observations = iter([poses[2], poses[3] + [1., 0., 0.]])
    monkeypatch.setattr('mgts_nav.recovery.estimate_initial', lambda *_: next(observations))
    for scan in scans[:3]:
        assert recovery.update(scan, False) is None
    if interruption == 'duplicate':
        scans[3].t = scans[2].t
    elif interruption == 'gap':
        scans[3].t += 2.
    assert recovery.update(scans[3], interruption == 'matched') is None
    assert recovery.update(scans[4], False) is None
    assert recovery.stats == {'attempts': 1, 'recoveries': 0}
