import io
from math import isnan

import pytest
from PIL import Image

from mgts_nav.quality import inspection_quality

from tools.plot_quality import Sample, plot


def test_plot_breaks_curve_at_gap_and_predictions():
    rows = [Sample(t, 'matched', .8) for t in (0., .1, .2, 1., 1.1, 1.2)]
    rows[4].status = 'prediction_only'
    fig, quality = plot(rows)
    ax = fig.axes[0]
    assert len(quality['scan_gaps']) == len(ax.patches) == 1
    assert len(ax.collections[0].get_offsets()) == 1
    x = ax.lines[0].get_xdata()
    assert list(x[:3]) == [0., .1, .2]
    assert isnan(x[3]) and x[4] == 1. and isnan(x[5]) and x[6] == 1.2


def test_single_scan_exports_png():
    fig, quality = plot([Sample(250., 'matched', .75)])
    assert quality['scan_gaps'] == []
    assert fig.axes[0].get_ylim() == (0., 1.)
    data = io.BytesIO()
    fig.savefig(data, format='png')
    with Image.open(data) as image:
        assert image.format == 'PNG'
        image.verify()


def test_two_scans_without_period_are_not_joined():
    fig, quality = plot([Sample(0., 'matched', .8), Sample(10., 'matched', .7)])
    assert quality['gap_threshold_s'] is None
    line = fig.axes[0].lines[0]
    x, y = list(line.get_xdata()), list(line.get_ydata())
    assert x[0] == 0. and x[-1] == 10.
    assert isnan(x[1]) and isnan(y[1])
    assert 'неизвестен' in fig.axes[0].get_xlabel()


def test_explicit_threshold_finds_gap_in_short_record():
    rows = [Sample(t, 'matched', .8) for t in (0., .1, 5.)]
    assert plot(rows)[1]['scan_gaps'] == []
    fig, quality = plot(rows, gap_threshold_s=.3)
    assert quality['gap_threshold_s'] == .3
    assert quality['scan_gaps'] == [dict(start_s=.1, end_s=5., duration_s=4.9)]
    assert len(fig.axes[0].patches) == 1
    x = list(fig.axes[0].lines[0].get_xdata())
    assert x[:2] == [0., .1] and isnan(x[2]) and x[3] == 5.


@pytest.mark.parametrize('threshold', [0., -1., float('nan'), float('inf')])
def test_invalid_gap_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match='Gap threshold'):
        inspection_quality([Sample(0., 'matched', .8), Sample(1., 'matched', .8)], threshold)
