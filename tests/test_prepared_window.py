import numpy as np
import pytest

from mgts_nav.bag import Scan
from mgts_nav import delayed_start as ds


@pytest.mark.parametrize("times,message", [
    (range(20), "Need at least three global search slots"),
    (range(21), "Gap in the local scan chain"),
])
def test_rejected_window_has_same_per_map_diagnostics(times, message):
    rows = [Scan(t, np.ones((12, 2)), np.zeros(2), None, None) for t in times]
    prepared = ds.prepare_window(iter(rows))
    for run in (lambda stats: ds.estimate_start(iter(rows), None, None, stats),
                lambda stats: ds.estimate_prepared(prepared, None, None, stats)):
        stats = {}
        with pytest.raises(ValueError, match=message):
            run(stats)
        assert stats == {"anchors": []}
