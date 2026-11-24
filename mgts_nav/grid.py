"""Sparse occupancy evidence, exported in ROS map_server PGM + YAML convention."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


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


PREVIEW_MIN_SIDE = 900
MATCHED, PREDICTED, START, FINISH = '#0074b8', '#dd7800', '#18803a', '#cc263b'


def _font(size):
    try:
        import matplotlib
        return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / 'fonts' / 'ttf' / 'DejaVuSans.ttf'), size)
    except (ImportError, OSError):
        try:
            return ImageFont.load_default(size)
        except TypeError:  # Pillow < 10.1 has only the small bitmap font
            return ImageFont.load_default()


def preview(path, output, trajectory):
    """Draw the route over the map, magnified with whole pixels, with legend and scale."""
    pixels, meta, _ = read_map(path)
    h, w = pixels.shape
    zoom = max(1, -(-PREVIEW_MIN_SIDE // max(h, w)))
    per_metre = zoom / meta['resolution']
    origin = np.array(meta['origin'][:2])
    coords = (np.array([r.pose[:2] for r in trajectory]) - origin) / meta['resolution']
    coords = np.column_stack([coords[:, 0], h - coords[:, 1]]) * zoom
    route = Image.fromarray(pixels).convert('RGB').resize((w * zoom, h * zoom), Image.NEAREST)
    draw = ImageDraw.Draw(route)
    line = max(2, zoom // 2)
    predicted = False
    for i in range(1, len(coords)):
        guess = trajectory[i].status == 'prediction_only'
        predicted |= guess
        draw.line([tuple(coords[i-1]), tuple(coords[i])], fill=PREDICTED if guess else MATCHED, width=line)
    dot = 2 * line + 2
    # The finish is a ring drawn under the start dot, so both stay visible on a closed loop.
    x, y = coords[-1]
    draw.ellipse([x - 2 * dot, y - 2 * dot, x + 2 * dot, y + 2 * dot], outline=FINISH, width=line + 1)
    x, y = coords[0]
    draw.ellipse([x - dot, y - dot, x + dot, y + dot], fill=START)

    size = max(16, route.width // 50)
    font = _font(size)
    items = [('start', 'старт'), ('finish', 'финиш'), ('line', 'маршрут по карте')]
    if predicted:
        items.append(('guess', 'только прогноз'))
    metres = next(m for m in (1, 2, 5, 10, 20, 50, 100, 200, 500) if m * per_metre >= 4 * size or m == 500)
    items.append(('scale', f'{metres} м'))
    measure = ImageDraw.Draw(route)
    widths = [2 * size + 8 + measure.textlength(text, font=font) + 2 * size if kind != 'scale'
              else metres * per_metre + 8 + measure.textlength(text, font=font) for kind, text in items]
    rows, row, used = [], [], 0
    for item, width in zip(items, widths):
        if row and used + width > route.width - size:
            rows.append(row)
            row, used = [], 0
        row.append((item, width))
        used += width
    rows.append(row)
    pad = size // 2
    footer = len(rows) * (size * 2) + pad
    image = Image.new('RGB', (route.width, route.height + footer), 'white')
    image.paste(route, (0, 0))
    draw = ImageDraw.Draw(image)
    for r, row in enumerate(rows):
        cy = route.height + pad + r * size * 2 + size // 2 + 2
        x = pad
        for (kind, text), width in row:
            mark = size // 2
            if kind == 'start':
                draw.ellipse([x + size - mark, cy - mark, x + size + mark, cy + mark], fill=START)
            elif kind == 'finish':
                draw.ellipse([x + size - mark, cy - mark, x + size + mark, cy + mark], outline=FINISH, width=3)
            elif kind in ('line', 'guess'):
                draw.line([(x, cy), (x + 2 * size, cy)], fill=MATCHED if kind == 'line' else PREDICTED, width=line)
            if kind == 'scale':
                bar = metres * per_metre
                draw.rectangle([x, cy - 3, x + bar, cy + 3], fill='black')
                draw.line([(x, cy - mark), (x, cy + mark)], fill='black', width=2)
                draw.line([(x + bar, cy - mark), (x + bar, cy + mark)], fill='black', width=2)
                draw.text((x + bar + 8, cy), text, fill='black', font=font, anchor='lm')
            else:
                draw.text((x + 2 * size + 8, cy), text, fill='black', font=font, anchor='lm')
            x += width
    image.save(output)
