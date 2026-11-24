import numpy as np
import pytest

from mgts_nav.observability import diagnose
from mgts_nav.slam import rotation


def test_parallel_walls_do_not_constrain_translation_along_corridor():
    x = np.linspace(-2., 2., 81)
    wall = np.column_stack((x, np.ones_like(x)))
    points = np.vstack((wall, wall - [0, 2]))
    for angle in (0., .62):
        transform = rotation(angle)
        report = diagnose(points @ transform.T + [12, -54])
        assert report['numerical_rank'] == 2 and report['condition_number'] is None
        assert report['normal_eigenvalues'][0] < 1e-10
        assert abs(np.dot(report['weakest_translation_xy'], transform[:, 0])) > .999


def test_corner_provides_three_constraints():
    x = np.linspace(0., 3., 61)
    wall = np.column_stack((x, np.zeros_like(x)))
    corner = np.vstack((wall, wall[:, ::-1]))
    result = diagnose(corner)
    assert result['numerical_rank'] == 3
    assert min(result['normal_eigenvalues']) > .4
    assert 1. < result['condition_number'] < 10.


def test_repeated_or_isolated_points_have_no_stable_normals():
    with pytest.raises(ValueError):
        diagnose(np.zeros((12, 2)))
    with pytest.raises(ValueError):
        diagnose(np.arange(20).reshape(-1, 2))
