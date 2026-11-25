import numpy as np
import pytest

from tools.tum_sparse_depth import BeamGrid, K


@pytest.mark.parametrize('channels', [16, 32])
def test_hidden_depth_cannot_supply_a_return(channels):
    grid = BeamGrid(channels)
    depth = np.full((480, 640), 15000, dtype=np.uint16)
    depth[grid.v, grid.u] = 0
    assert len(grid.observe(depth).points) == 0
    i = np.argmin(np.linalg.norm(grid.angles - [5., 5.], axis=1))
    depth[grid.v[i], grid.u[i]] = 10000
    view = grid.observe(depth)
    np.testing.assert_allclose(view.points, [2*grid.rays[i]])
    x, y, _ = grid.rays[i]
    u, v = K[0, 0]*x + K[0, 2], K[1, 1]*y + K[1, 2]
    assert view.find(u, v) == 0
    assert view.find(u + 10, v) is None
    tracks = [[identity, u, v, u, v] for identity in (0, 1)]
    a, b = grid.pairs(tracks, [depth, depth], None)
    assert len(a) == len(b) == 1
    np.testing.assert_equal(a, b)
