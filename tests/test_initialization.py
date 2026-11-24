import numpy as np
import pytest

from mgts_nav.initialization import estimate_initial
from mgts_nav.slam import rotation


def scene(asymmetric):
    pixels = np.full((41, 61), 254, dtype=np.uint8)
    pixels[[0, -1], :] = 0
    pixels[:, [0, -1]] = 0
    if asymmetric:
        pixels[8:21, 12] = 0
        pixels[20, 12:28] = 0
    meta = dict(origin=[-2., -2., 0.], resolution=.1, occupied_thresh=.65, free_thresh=.196)
    walls = (np.argwhere(pixels[::-1] == 0)[:, ::-1] + .5)*.1 - 2
    pose = np.array([.56, -.27, .38])
    return (walls - pose[:2]) @ rotation(pose[2]), pixels, meta, pose


def test_global_pose_without_prior():
    xy, pixels, meta, pose = scene(True)
    np.testing.assert_allclose(estimate_initial(xy, pixels, meta), pose, atol=.09)


def test_rotational_symmetry_rejected():
    xy, pixels, meta, _ = scene(False)
    with pytest.raises(ValueError, match='Ambiguous'):
        estimate_initial(xy, pixels, meta)


def test_sparse_scan_rejected():
    xy, pixels, meta, _ = scene(True)
    with pytest.raises(ValueError):
        estimate_initial(xy[:5], pixels, meta)
