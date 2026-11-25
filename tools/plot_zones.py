"""Render the inspection route and zone boundaries from a completed solve."""
import argparse
from collections import Counter
import csv
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from PIL import Image

from mgts_nav.grid import read_map
from mgts_nav.zones import load_zones


def plot(solution):
    pixels, meta, _ = read_map(solution / 'map/map.yaml')
    zones = load_zones(solution / 'profiles/zones.json')
    with (solution / 'inspection/zone-visits.csv').open(encoding='utf-8-sig', newline='') as source:
        rows = csv.DictReader(source)
        if 'zone' not in (rows.fieldnames or []):
            raise ValueError('Missing zone column in visit report')
        counts = Counter(row['zone'] for row in rows)
    if counts.keys() - zones.keys():
        raise ValueError('Visit report contains an unknown zone')
    fig = Figure(figsize=(8, 6), layout='constrained')
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    x, y = meta['origin'][:2]
    h, w = pixels.shape
    r = meta['resolution']
    with Image.open(solution / 'inspection/route.png') as route:
        if route.size != (w, h):
            raise ValueError('Route image dimensions differ from map')
        ax.imshow(route.convert('RGB'), origin='upper', interpolation='nearest',
                  extent=(x, x + w*r, y, y + h*r))
    for i, (name, vertices) in enumerate(zones.items()):
        ax.add_patch(Polygon(vertices, fill=False, edgecolor=f'C{(i+4)%10}', linewidth=2,
                             linestyle='-' if counts[name] else '--',
                             label=f'{name}   ·   {counts[name]}', clip_on=False))
    ax.autoscale_view()
    ax.set(xlabel='X карты, м', ylabel='Y карты, м', aspect='equal',
           title='Зоны на маршруте инспекции')
    ax.legend(title='Зона · подтверждённые интервалы', loc='upper left',
              bbox_to_anchor=(1, 1), frameon=False)
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('solution', type=Path)
    parser.add_argument('png', type=Path)
    args = parser.parse_args()
    try:
        figure = plot(args.solution)
        with args.png.open('xb') as output:
            figure.savefig(output, format='png', dpi=150)
    except (OSError, ValueError) as error:
        parser.exit(2, f'{error}\n')
