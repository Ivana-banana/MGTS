import numpy as np
import pytest

from mgts_nav import coarse_start
from mgts_nav.bag import Scan


def test_repeated_returns_do_not_supply_geometry():
    scan = Scan(1., np.ones((100, 2)), np.zeros(2), None, None)
    with pytest.raises(ValueError, match="Too few"):
        coarse_start.estimate(scan, np.zeros((40, 40), np.uint8),
                              {"resolution": .08, "origin": [0., 0., 0.]}, {})


def test_expired_budget_abstains_before_search(monkeypatch):
    monkeypatch.setattr(coarse_start, "TIME_LIMIT_S", 0.)
    pixels = np.full((40, 40), 254, dtype=np.uint8)
    pixels[:2, :] = 0
    scan = Scan(1., np.column_stack((np.linspace(-1, 1, 50), np.ones(50))), np.zeros(2), None, None)
    meta = {"resolution": .08, "origin": [0., 0., 0.], "occupied_thresh": .65, "free_thresh": .196}
    stats = {}
    with pytest.raises(ValueError, match="budget"):
        coarse_start.estimate(scan, pixels, meta, stats)
    assert (stats["coarse"], stats["fine"], stats["icp"]) == (0, 0, 0)
