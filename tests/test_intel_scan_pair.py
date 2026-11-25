import numpy as np

from tools.intel_scan_pair import search


def test_asymmetric_geometry_recovers_nonzero_pose():
    first = np.random.default_rng(115).uniform([-2., -1.], [3., 2.], (100, 2))
    # Sensor faces +Y, so a first-frame point maps to (y - ty, tx - x).
    current = np.column_stack((first[:, 1] + .4, .8 - first[:, 0]))
    result = search(first, current)
    assert not result["refusals"]
    np.testing.assert_allclose(result["pose"], [.8, -.4, np.pi/2], atol=1e-8)


def test_symmetric_geometry_refuses_half_turn():
    half = np.random.default_rng(114).uniform([.1, -.9], [1.8, .9], (34, 2))
    cloud = np.vstack([half, -half])
    result = search(cloud, cloud)
    assert result["refusals"] == ["ambiguous"]
    assert result["score"] < 1e-10 and result["rival_score"] < 1e-10
    assert result["evaluations"] <= 33912
