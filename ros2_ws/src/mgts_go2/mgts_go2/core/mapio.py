"""Map files: binary PCD (x y z intensity), 2D occupancy slice (PGM + YAML), pose CSV."""
import csv
import json
from pathlib import Path

import numpy as np

from . import se3


def write_pcd(path, points, intensity=None):
    points = np.asarray(points, dtype=np.float32)
    n = len(points)
    if intensity is None:
        intensity = np.zeros(n, dtype=np.float32)
    data = np.empty(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4')])
    data['x'], data['y'], data['z'] = points[:, 0], points[:, 1], points[:, 2]
    data['intensity'] = np.asarray(intensity, dtype=np.float32)
    header = ('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\n'
              'SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\n'
              'VIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA binary\n' % (n, n))
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())


def read_pcd(path):
    """Read binary or ascii PCD with float x y z [intensity]. Returns (points, intensity or None)."""
    raw = Path(path).read_bytes()
    pos = 0
    hdr = {}
    while True:
        end = raw.index(b'\n', pos)
        line = raw[pos:end].decode('ascii').strip()
        pos = end + 1
        if not line or line.startswith('#'):
            continue
        key, _, val = line.partition(' ')
        hdr[key.upper()] = val.split()
        if key.upper() == 'DATA':
            break
    fields = hdr['FIELDS']
    sizes = [int(s) for s in hdr['SIZE']]
    types = hdr['TYPE']
    counts = [int(c) for c in hdr.get('COUNT', ['1'] * len(fields))]
    n = int(hdr['POINTS'][0])
    kind = {('F', 4): '<f4', ('F', 8): '<f8', ('U', 1): 'u1', ('U', 2): '<u2', ('U', 4): '<u4',
            ('I', 1): 'i1', ('I', 2): '<i2', ('I', 4): '<i4'}
    if hdr['DATA'][0] == 'binary':
        dt = np.dtype([(f if c == 1 else f, kind[(t, s)], (c,) if c > 1 else ())
                       for f, s, t, c in zip(fields, sizes, types, counts)])
        arr = np.frombuffer(raw, dtype=dt, count=n, offset=pos)
        get = lambda name: arr[name].astype(np.float64)
    elif hdr['DATA'][0] == 'ascii':
        table = np.loadtxt(raw[pos:].decode('ascii').splitlines(), ndmin=2)
        get = lambda name: table[:, fields.index(name)]
    else:
        raise ValueError('unsupported PCD DATA %s' % hdr['DATA'])
    pts = np.column_stack([get('x'), get('y'), get('z')])
    inten = get('intensity') if 'intensity' in fields else None
    return pts, inten


def write_grid(stem, points, resolution=0.05, z_min=0.15, z_max=1.2, floor_z=0.0, padding=1.0):
    """2D occupancy slice for nav2 map_server: occupied where map points lie in the height band."""
    rel = points[:, 2] - floor_z
    sel = points[(rel >= z_min) & (rel <= z_max)]
    lo = points[:, :2].min(axis=0) - padding
    hi = points[:, :2].max(axis=0) + padding
    w, h = (np.ceil((hi - lo) / resolution).astype(int) + 1)
    img = np.full((h, w), 205, dtype=np.uint8)  # unknown
    # free: cells covered by the floor (points near floor height)
    floor = points[np.abs(rel) < 0.1]
    for pts, val in ((floor, 254), (sel, 0)):
        ij = np.floor((pts[:, :2] - lo) / resolution).astype(int)
        img[h - 1 - ij[:, 1], ij[:, 0]] = val
    stem = Path(stem)
    with open(stem.with_suffix('.pgm'), 'wb') as f:
        f.write(b'P5\n%d %d\n255\n' % (w, h))
        f.write(img.tobytes())
    stem.with_suffix('.yaml').write_text(
        'image: %s\nresolution: %.3f\norigin: [%.3f, %.3f, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n'
        % (stem.with_suffix('.pgm').name, resolution, lo[0], lo[1]), encoding='utf-8')
    return img


def write_poses(path, stamps, poses, extra=None, extra_names=()):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['stamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', *extra_names])
        for i, (s, t) in enumerate(zip(stamps, poses)):
            q = se3.rot_to_quat(t[:3, :3])
            row = ['%.9f' % s] + ['%.6f' % v for v in t[:3, 3]] + ['%.8f' % v for v in q]
            if extra is not None:
                row += list(extra[i])
            w.writerow(row)


def read_poses(path):
    stamps, poses = [], []
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            stamps.append(float(row['stamp']))
            poses.append(se3.from_pose((float(row['x']), float(row['y']), float(row['z'])),
                                       (float(row['qx']), float(row['qy']), float(row['qz']), float(row['qw']))))
    return np.array(stamps), poses


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
