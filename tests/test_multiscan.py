import numpy as np
import pytest

from mgts_nav.bag import Scan
from mgts_nav.initialization import estimate_initial
from mgts_nav.multiscan import MAX_SCANS, _registered_prefix, estimate_multiscan
from mgts_nav.slam import rotation


def simulate(asymmetric, fov):
    """Ray-march the first occupied cell, as viewed through a limited aperture."""
    pixels = np.full((41, 61), 254, dtype=np.uint8)
    pixels[[0, -1], :] = 0
    pixels[:, [0, -1]] = 0
    if asymmetric:
        pixels[8:21, 12] = 0
        pixels[20, 12:28] = 0
    meta = dict(origin=[-2., -2., 0.], resolution=.1, occupied_thresh=.65, free_thresh=.196)
    pose = np.array([.56, -.27, .38])
    scans = []
    ranges = np.arange(.05, 8, .025)
    for t in np.linspace(0, 3.0, 31):
        state = pose + np.array([.25 * t, -.12 * t, 1.7 * t])
        ray = state[2] + np.deg2rad(np.linspace(-fov/2, fov/2, 128))
        directions = np.column_stack((np.cos(ray), np.sin(ray)))
        points = state[:2] + directions[:, None, :] * ranges[None, :, None]
        ix, iy = np.floor((points + 2.0) / .1).astype(int).transpose(2, 0, 1)
        occupied = pixels[::-1][iy.clip(0, 40), ix.clip(0, 60)] == 0
        assert occupied.any(axis=1).all()
        j = occupied.argmax(axis=1)
        xy = directions * ranges[j, None] @ rotation(state[2])
        # Map and odom origins/yaws differ, so using absolute odometry will fail.
        odom = np.r_[rotation(.7) @ state[:2] + [10., -7.], state[2] + .7]
        scans.append(Scan(t, xy, np.zeros(2), odom, None))
    return scans, pixels, meta, pose


@pytest.mark.parametrize("asymmetric,fov", [(True, 20), (True, 40), (True, 80),
                                             (True, 360), (False, 80)])
def test_starts_by_visibility_and_symmetry(asymmetric, fov):
    scans, pixels, meta, truth = simulate(asymmetric, fov)

    def outcome(predict):
        try:
            p = predict()
        except ValueError:
            return "refused"
        xy = np.linalg.norm(p[:2] - truth[:2])
        yaw = abs(np.arctan2(np.sin(p[2] - truth[2]), np.cos(p[2] - truth[2])))
        return "correct" if xy < 0.17 and yaw < 0.12 else "wrong"

    one = outcome(lambda: estimate_initial(scans[0].xy, pixels, meta))
    multi = outcome(lambda: estimate_multiscan(scans, pixels, meta))
    print(f"asymmetric={asymmetric} FOV={fov}: single={one}, multi={multi}")
    assert multi == ("correct" if asymmetric else "refused")
    assert one == ("correct" if asymmetric and fov >= 80 else "refused")


def test_window_budget_and_invalid_odometry():
    xy = np.arange(60).reshape(-1, 2)

    def too_long():
        for i in range(MAX_SCANS):
            yield Scan(i / 100, xy, np.zeros(2), np.zeros(3), None)
        pytest.fail("Consumed beyond maximum number of scans")

    cloud = _registered_prefix(too_long())
    np.testing.assert_array_equal(cloud, xy)
    first = Scan(1, xy, np.zeros(2), np.zeros(3), None)
    with pytest.raises(ValueError, match="odometry"):
        _registered_prefix([first, Scan(2, xy, np.zeros(2), None, None)])
    with pytest.raises(ValueError, match="two"):
        _registered_prefix([first, Scan(4.01, xy, np.zeros(2), None, None)])
    meta = dict(resolution=.1)
    with pytest.raises(ValueError, match="budget"):
        estimate_multiscan([], np.broadcast_to(np.uint8(0), (500, 500)), meta)
