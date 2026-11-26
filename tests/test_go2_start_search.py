import numpy as np
import pytest

from tools.go2_start_search import placed, search


@pytest.mark.parametrize('duplicate', [False, True])
def test_known_start_and_repeated_geometry(duplicate):
    rng = np.random.default_rng(42)
    points = rng.uniform([-2., -1., .15], [1., 2., 1.3], (150, 3))
    expected = np.array([2., -1., np.deg2rad(40)])
    fixed = placed(points, expected)
    if duplicate:
        fixed = np.vstack((fixed, fixed + [6., 0., 0.]))
    best, score, (rival_score, rival) = search(fixed, points, np.zeros(3))
    assert score < .003
    if duplicate:
        assert rival_score - score < .003
        assert np.linalg.norm(best[:2] - rival[:2]) > 5.9
    else:
        np.testing.assert_allclose(best, expected, atol=.003)
        assert rival_score - score > .02
