import json

from PIL import Image

from mgts_nav.grid import Grid
from tools.plot_zones import plot


def test_empty_zone_at_top_edge_of_metric_map(tmp_path):
    for folder in ('map', 'profiles', 'inspection'):
        (tmp_path / folder).mkdir()
    grid = Grid(.5)
    grid.evidence.update({(x, 4): 6 for x in range(-4, 8)})
    grid.save(tmp_path / 'map')
    with Image.open(tmp_path / 'map/map.pgm') as pgm:
        route = pgm.convert('RGB')
    route.putpixel((0, 0), (12, 34, 56))
    route.save(tmp_path / 'inspection/route.png')
    vertices = [[-3, 3.5], [-2, 3.5], [-2, 4]]
    (tmp_path / 'profiles/zones.json').write_text(json.dumps({
        'frame': 'map', 'zones': {'upper': vertices}}))
    (tmp_path / 'inspection/zone-visits.csv').write_text('zone,first_inside_t\n')
    ax = plot(tmp_path).axes[0]
    raster = ax.images[0]
    assert tuple(raster.get_extent()) == (-3, 5, 1, 3.5)
    assert raster.origin == 'upper' and tuple(raster.get_array()[0, 0]) == (12, 34, 56)
    assert ax.get_ylim()[1] >= 4
    assert ax.patches[0].get_xy()[:3].tolist() == vertices
    assert ax.patches[0].get_label().endswith('0') and ax.patches[0].get_linestyle() == '--'
