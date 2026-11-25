"""Try delayed initialization against every 40-scan section of a field survey."""
import argparse
import csv
from itertools import islice
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mgts_nav.bag import load_config, scans
from mgts_nav.delayed_start import estimate_prepared, prepare_window
from mgts_nav.grid import Grid, read_map
from mgts_nav.native_grid import NativeGrid, load_dll
from mgts_nav.slam import world


def search(data, full_map, output, dll=None):
    profile = load_config(data/'profile.json')
    _, meta, _ = read_map(full_map/'map.yaml')
    prepared = prepare_window(scans(data/'inspection', profile))
    backend = load_dll(dll) if dll else None
    sections, accepted = [], []
    with (full_map/'trajectory.csv').open(encoding='utf-8-sig', newline='') as source:
        rows = zip(scans(data/'mapping', profile), csv.DictReader(source), strict=True)
        first = 0
        while chunk := list(islice(rows, 40)):
            grid = Grid(meta['resolution']) if backend is None else NativeGrid(meta['resolution'], backend)
            for scan, row in chunk:
                if scan.t != float(row['t']):
                    raise ValueError('Survey scan timestamp differs from trajectory')
                if row['status'] == 'prediction_only':
                    continue
                if row['status'] not in ('matched', 'origin'):
                    raise ValueError(f"Unknown survey status {row['status']!r}")
                pose = np.array([float(row[k]) for k in ('x', 'y', 'yaw')])
                grid.observe(world(scan.origin, pose), world(scan.xy, pose))
            section = output/f'section-{first}'
            section.mkdir()
            report = dict(first_scan=first, scans=len(chunk))
            stats = {}
            try:
                grid.save(section)
                pixels, submeta, _ = read_map(section/'map.yaml')
                start = estimate_prepared(prepared, pixels, submeta, stats)
            except ValueError as exc:
                report['reason'] = str(exc)
            else:
                report['start'] = start.tolist()
                accepted.append(report['start'])
            report['anchors'] = stats.get('anchors', [])
            sections.append(report)
            first += len(chunk)
    if first != 300:
        raise ValueError('Expected 300 survey scans')
    result = dict(sections=sections, accepted_sections=len(accepted))
    if len(accepted) == 1:
        (output/'start.json').write_text(json.dumps(accepted[0]), encoding='utf-8')
    (output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('data', 'full_map', 'output'):
        parser.add_argument(arg, type=Path)
    parser.add_argument('--grid-dll', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        stage = Path(tmp)/'result'
        stage.mkdir()
        report = search(args.data, args.full_map, stage, args.grid_dll)
        stage.rename(args.output)
    print(f"Accepted sections: {report['accepted_sections']} of {len(report['sections'])}")
