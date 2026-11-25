import numpy as np
from PIL import Image

from mgts_nav.grid import Grid, preview
from mgts_nav.quality import inspection_quality
from mgts_nav.slam import Estimate


def test_preview_breaks_line_at_reported_gap_and_keeps_isolated_scan(tmp_path):
    grid = Grid(1)
    grid.evidence.update({(x, 0): 6 for x in range(41)})
    grid.evidence[(0, 10)] = -1
    grid.save(tmp_path)
    times = (0, 1, 2, 10, 20, 21, 22)
    xs = (1, 4, 7, 17, 27, 32, 38)
    rows = [Estimate(t, np.array([x, 5, 0]), 'matched', 1.) for t, x in zip(times, xs)]
    rows[3].status = 'prediction_only'
    assert len(inspection_quality(rows)['scan_gaps']) == 2
    preview(tmp_path / 'map.yaml', tmp_path / 'route.png', rows)
    with Image.open(tmp_path / 'route.png') as route:
        # Grid.save pads the map on every side by two cells.
        pixel = lambda x: route.getpixel((x + 2, 7))
        assert pixel(5) == pixel(30) == (0, 116, 184)
        assert pixel(12) == pixel(22) == (205, 205, 205)
        assert pixel(17) == (221, 120, 0)
