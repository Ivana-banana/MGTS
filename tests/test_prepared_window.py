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
        assert stats['anchors'] == []
        if len(rows) >= 21:
            assert stats['rejected_pair'] == dict(indices=[0, 1], times_s=[0, 1], interval_s=1,
                                                  points=[12, 12], causes=['time_interval'])
        else:
            assert 'rejected_pair' not in stats


@pytest.mark.parametrize('delay,causes', [(0, ['too_few_points']), (1, ['time_interval', 'too_few_points'])])
def test_sparse_scan_at_first_bad_pair(delay, causes):
    rows = [Scan(i*.1 + (delay if i >= 7 else 0), np.ones((11 if i == 7 else 20, 2)),
                 np.zeros(2), None, None) for i in range(25)]
    stats = {}
    with pytest.raises(ValueError, match='Gap in the local scan chain'):
        ds.estimate_start(rows, None, None, stats)
    assert stats['rejected_pair']['indices'] == [6, 7]
    assert stats['rejected_pair']['points'] == [20, 11]
    assert stats['rejected_pair']['causes'] == causes
