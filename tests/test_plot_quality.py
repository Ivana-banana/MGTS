import io
from math import isnan

from PIL import Image

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
