"""Sparse occupancy evidence, exported in ROS map_server PGM + YAML convention."""
from collections import defaultdict
from math import isfinite
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw


DEFAULT_RESOLUTION = 0.08


def check_resolution(resolution):
    if not isfinite(resolution) or not 0.01 <= resolution <= 1:
        raise ValueError('resolution must be between 0.01 and 1 metre')
    return resolution


class Grid:
    def __init__(self, resolution):
        self.resolution = check_resolution(resolution)
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
    try:
        meta = yaml.safe_load(path.read_text(encoding='utf-8'))
    except yaml.YAMLError as exc:
        raise ValueError(f'{path}: invalid map YAML: {exc}') from exc
    if not isinstance(meta, dict):
        raise ValueError(f'{path}: expected a YAML mapping of map fields')

    def finite_number(value):
        try:
            return type(value) in (float, int) and isfinite(value)
        except OverflowError:
            return False

    coords = meta.get('origin')
    if not isinstance(coords, list) or len(coords) != 3 or not all(map(finite_number, coords)):
        raise ValueError(f'{path}: origin must contain three finite numbers [x, y, yaw]')
    if coords[2] != 0:
        raise ValueError(f'{path}: origin yaw must be zero for an axis-aligned map')
    if meta.get('negate', 0) != 0:
        raise ValueError(f'{path}: negate must be zero')
    resolution = meta.get('resolution')
    if not finite_number(resolution) or resolution <= 0:
        raise ValueError(f'{path}: resolution must be finite and positive')
    for field in ('free_thresh', 'occupied_thresh'):
        threshold = meta.get(field)
        if not finite_number(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f'{path}: {field} must be a finite probability between 0 and 1')
    if meta['free_thresh'] >= meta['occupied_thresh']:
        raise ValueError(f'{path}: free_thresh must be less than occupied_thresh')
    if not isinstance(meta.get('image'), str) or not meta['image'].strip():
        raise ValueError(f'{path}: image must name a map image file')
    with Image.open(path.parent / meta['image']) as image:
        pixels = np.array(image.convert('L'))
    origin = np.asarray(meta['origin'], dtype=float)
    y, x = np.where((1.0 - pixels / 255.0) > meta['occupied_thresh'])
    with np.errstate(over='ignore'):
        points = origin[:2] + (np.column_stack([x, len(pixels) - 1 - y]) + 0.5) * resolution
    if not np.isfinite(points).all():
        raise ValueError(f'{path}: origin and resolution produce non-finite map coordinates')
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
