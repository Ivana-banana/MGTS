"""Sparse occupancy evidence, exported in ROS map_server PGM + YAML convention."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw


class Grid:
    def __init__(self, resolution):
        if not np.isfinite(resolution) or not 0.01 <= resolution <= 1:
            raise ValueError('resolution must be between 0.01 and 1 metre')
        self.resolution = resolution
        self.evidence = defaultdict(int)

    def observe(self, origin, endpoints):
        start = np.floor(origin / self.resolution).astype(int)
        # Discard distant returns before rasterizing improbable rays.
        endpoints = endpoints[np.linalg.norm(endpoints - origin, axis=1) <= 35]
        cells = np.unique(np.floor(endpoints / self.resolution).astype(int), axis=0)
        free = set()
        occupied = set(map(tuple, cells))
        for end in cells:
            delta = end - start
            count = int(abs(delta).max())
            if count:
                samples = start + np.rint(np.arange(count)[:, None] * delta / count).astype(int)
                free.update(map(tuple, samples))
        for cell in free - occupied:
            self.evidence[cell] = max(-12, self.evidence[cell] - 1)
        for cell in occupied:
            self.evidence[cell] = min(12, self.evidence[cell] + 3)

    def save(self, directory):
        if not self.evidence:
            raise ValueError('No usable scan contributed to the map')
        cells = np.array(list(self.evidence))
        low, high = cells.min(axis=0) - 2, cells.max(axis=0) + 2
        w, h = high - low + 1
        if int(w) * int(h) > 60_000_000:
            raise ValueError('Map extent is implausible; check units and calibration')
        pixels = np.full((h, w), 205, dtype=np.uint8)
        x, y = (cells - low).T
        votes = np.array(list(self.evidence.values()))
        pixels[h - 1 - y, x] = np.where(votes >= 5, 0, np.where(votes < 0, 254, 205))
        if (pixels == 0).sum() < 12:
            raise ValueError('Too few repeatedly observed occupied cells')
        Image.fromarray(pixels).save(directory / 'map.pgm')
        meta = dict(image='map.pgm', resolution=float(self.resolution),
                    origin=[*(low * self.resolution).tolist(), 0.0], negate=0,
                    occupied_thresh=0.65, free_thresh=0.196)
        (directory / 'map.yaml').write_text(yaml.safe_dump(meta), encoding='utf-8')


def read_map(path):
    path = Path(path)
    meta = yaml.safe_load(path.read_text(encoding='utf-8'))
    with Image.open(path.parent / meta['image']) as image:
        pixels = np.array(image.convert('L'))
    origin = np.asarray(meta['origin'], dtype=float)
    if origin[2] != 0 or meta.get('negate', 0) != 0:
        raise ValueError('Expected an axis-aligned, non-negated MGTS map')
    y, x = np.where((1.0 - pixels / 255.0) > meta['occupied_thresh'])
    points = origin[:2] + (np.column_stack([x, len(pixels) - 1 - y]) + 0.5) * meta['resolution']
    if len(points) < 12:
        raise ValueError('Map has too few occupied cells')
    return pixels, meta, points


def preview(path, output, trajectory):
    pixels, meta, _ = read_map(path)
    origin = np.array(meta['origin'][:2])
    coords = (np.array([r.pose[:2] for r in trajectory]) - origin) / meta['resolution']
    coords[:, 1] = len(pixels) - 1 - coords[:, 1]
    image = Image.fromarray(pixels).convert('RGB')
    draw = ImageDraw.Draw(image)
    for i, p in enumerate(coords[1:], 1):
        color = '#dd7800' if trajectory[i].status == 'prediction_only' else '#0074b8'
        draw.line([tuple(coords[i-1]), tuple(p)], fill=color, width=2)
    for p, color in [(coords[0], '#18803a'), (coords[-1], '#cc263b')]:
        x, y = p
        draw.ellipse([x-2, y-2, x+2, y+2], fill=color)
    image.save(output)
