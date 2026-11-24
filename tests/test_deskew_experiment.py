import numpy as np
import pytest

from mgts_nav.bag import scans
from mgts_nav.demo import START_NS, generate
from mgts_nav.metrics import read_csv
from tools.evaluate_deskew import scan_wall_rmse, sensor_profile, write_bag


@pytest.fixture(scope='module')
def recordings(tmp_path_factory):
    root = tmp_path_factory.mktemp('sweep')
    original = root / 'original'
    generate(original, seed=41)
    profile = sensor_profile()
    for name, options in [('ns', {}), ('us', dict(unit='us')), ('reversed', dict(sign=-1)),
                          ('gap', dict(odom_gap=True))]:
        write_bag(original / 'mapping', root / name, 'mapping', 41, profile, **options)
    return root


def corrected(unit='ns'):
    return dict(sensor_profile(), point_time=dict(field='offset', unit=unit))


def test_loader_passes_only_the_bracketing_odometry(recordings, monkeypatch):
    from mgts_nav import bag

    original = bag.deskew_points
    counts = []

    def observe(xyz, elapsed, t, odom, gap):
        counts.append(len(odom))
        lo = t + min(0., elapsed.min())
        hi = t + max(0., elapsed.max())
        assert odom[0, 0] <= lo < odom[1, 0]
        assert odom[-2, 0] < hi <= odom[-1, 0]
        return original(xyz, elapsed, t, odom, gap)

    monkeypatch.setattr(bag, 'deskew_points', observe)
    rows = list(scans(recordings / 'ns', corrected()))
    assert len(rows) == 200 and len(counts) > 190


def test_signed_offsets_use_the_declared_units_and_sensor_extrinsics(recordings):
    nano = list(scans(recordings / 'ns', corrected()))
    micro = list(scans(recordings / 'us', corrected('us')))
    off = list(scans(recordings / 'ns', sensor_profile()))
    for a, b in zip(nano, micro, strict=True):
        assert a.t == b.t
        np.testing.assert_allclose(a.xy, b.xy, atol=1e-8, rtol=0)
    # All rays in this initial sweep were acquired while the body was stationary.
    np.testing.assert_allclose(nano[4].xy, off[4].xy, atol=1e-8, rtol=0)
    truth = read_csv(recordings / 'original/mapping_truth.csv')
    def moving_error(name, config):
        return scan_wall_rmse(recordings / name, config, 'mapping', truth)['moving']
    raw = moving_error('ns', sensor_profile())
    assert moving_error('ns', corrected()) < raw
    assert moving_error('reversed', corrected()) > raw


def test_gap_overlaps_late_rays_even_when_header_has_odometry(recordings):
    cfg = corrected()
    rows = list(scans(recordings / 'gap', cfg))
    start = START_NS['mapping'] / 1e9
    def at(offset):
        return next(r for r in rows if abs(r.t - start - offset) < 1e-8)
    assert len(at(3.935).xy) > 12
    assert at(4.035).odom is not None
    assert len(at(4.035).xy) == 0
    assert len(at(4.635).xy) > 12
