"""Generate paired known-start HEAP-1 tracking inputs for the support warning experiment."""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from mgts_nav.bag import load_config, scans
from mgts_nav.grid import read_map
from mgts_nav.metrics import Pose, read_csv
from mgts_nav.slam import track


def windows(data, first, count, expected_start):
    inspection = list(scans(data / 'inspection', load_config(data / 'profile.json')))
    truth = read_csv(data / 'inspection-reference.csv')
    if len(inspection) != count or len(truth) != count or abs(truth[0].t - expected_start) > 1e-6:
        raise ValueError(f'{data}: expected {count} samples starting at {expected_start}')
    if any(abs(s.t - p.t) > 1e-6 for s, p in zip(inspection, truth)):
        raise ValueError(f'{data}: scan and reference timestamps disagree')
    for offset in range(0, count, 60):
        window = inspection[offset:offset + 60]
        window[0] = replace(window[0], gyro_delta=None)
        yield first + offset, window, truth[offset:offset + 60]


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def generate(args):
    maps = {name: read_map(path)[2] for name, path in [('short', args.short_map), ('long', args.long_map)]}
    args.output.mkdir(parents=True, exist_ok=False)
    index = []
    for batch in (windows(args.initial, 2560, 60, 1731588668.081259),
                  windows(args.heldout, 2620, 180, 1731588674.081123)):
        for first, window, truth in batch:
            reference = Path('truth') / f'{first}.csv'
            write_csv(args.output / reference, Pose._fields, truth)
            for name, points in maps.items():
                rows = list(track(window, np.array(truth[0][1:]), map_points=points))
                if [r.t for r in rows] != [s.t for s in window]:
                    raise ValueError(f'{name}/{first}: incomplete tracking output')
                trajectory = Path(name) / str(first) / 'trajectory.csv'
                write_csv(args.output / trajectory, ['t', 'x', 'y', 'yaw', 'status', 'inlier_fraction'],
                          ([r.t, *r.pose, r.status, r.inliers] for r in rows))
                index.append(dict(map=name, first=first, trajectory=trajectory.as_posix(), truth=reference.as_posix()))
    (args.output / 'index.json').write_text(json.dumps(index, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, default=Path('out/heap1-long-80'))
    parser.add_argument('--heldout', type=Path, default=Path('out/heap1-heldout-37'))
    parser.add_argument('--short-map', type=Path, default=Path('out/heap1-revisit-map/map.yaml'))
    parser.add_argument('--long-map', type=Path, default=Path('out/heap1-long-score-80-repeat/map/map.yaml'))
    parser.add_argument('--output', type=Path, default=Path('out/issue160-inputs'))
    generate(parser.parse_args())
