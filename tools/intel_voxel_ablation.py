"""Field ablation of input and keyframe voxel reduction on the fixed Intel Lab windows."""
import argparse
from collections import Counter, deque
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from mgts_nav import slam
from mgts_nav.bag import load_config, scans
from mgts_nav.grid import Grid, read_map
from mgts_nav.metrics import read_csv
from tools.intel_laserscan import trajectory
from tools.intel_order_odom import compare_epochs
from tools.intel_scan_divergence import raw_scans


def reduce_points(points, mode):
    if mode == 'all':
        return points
    cells = np.floor(points / .06)
    if mode == 'first':
        return points[np.unique(cells, axis=0, return_index=True)[1]]
    _, groups, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    total = np.zeros((len(counts), 2))
    np.add.at(total, groups, points)
    return total / counts[:, None]


def mapping(stream, grid, mode):
    # Same keyframe insertion and motion model as slam.track; vary how the local tree is reduced.
    frames = deque(maxlen=18)
    pose = np.zeros(3)
    keypose = previous = tree = None
    for scan in stream:
        pose = slam.predict(pose, previous, scan)
        previous = scan.odom
        status, support = 'prediction_only', 0.
        if tree is None and len(scan.xy) >= slam.MIN_MATCH_POINTS:
            status = 'origin'
        elif tree is not None:
            pose, support, ok = slam.match(scan.xy, tree, pose)
            if ok:
                status = 'matched'
        if status != 'prediction_only':
            points = slam.world(scan.xy, pose)
            grid.observe(slam.world(scan.origin, pose), points)
            if (keypose is None or np.linalg.norm(pose[:2]-keypose[:2]) >= .15
                    or abs(pose[2]-keypose[2]) >= .12):
                frames.append(points)
                tree = cKDTree(reduce_points(np.vstack(frames), mode))
                keypose = pose.copy()
        yield slam.Estimate(scan.t, pose.copy(), status, support)


def trial(data, comparison, out, mode, label):
    cfg = load_config(data/'profile.json')
    source = data if label == 'direct' else comparison
    result = {}
    grid = Grid(.08)
    for role, product, count in [('mapping', 'map', 824), ('inspection', 'known', 60)]:
        path = source/role
        original = list(scans(path, cfg))
        raw = raw_scans(path, range(count))
        if len(original) != count or len(raw) != count:
            raise ValueError(f'Expected the {count}-scan Intel Lab window')
        for i, scan in enumerate(original):
            np.testing.assert_array_equal(reduce_points(raw[i], 'first'), scan.xy)
        start = perf_counter()
        stream = [replace(s, xy=reduce_points(raw[i], mode)) for i, s in enumerate(original)]
        directory = out/product
        directory.mkdir(parents=True)
        if role == 'mapping':
            rows = list(mapping(stream, grid, mode))
            grid.save(directory)
        else:
            _, _, points = read_map(out/'map/map.yaml')
            pose = slam.read_initial_pose(data/'external-start.json')
            rows = list(slam.track(stream, pose, map_points=points))
        elapsed = perf_counter()-start
        result_csv = directory/'trajectory.csv'
        with result_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(('t', 'x', 'y', 'yaw', 'status', 'inlier_fraction'))
            writer.writerows((r.t, *r.pose, r.status, r.inliers) for r in rows)
        result[product] = dict(seconds=elapsed, points=sum(len(s.xy) for s in stream),
                               status=Counter(r.status for r in rows),
                               rbpf=compare_epochs(result_csv, read_csv(data/f'{role}-external.csv')))
        if mode == 'first':
            names = ['trajectory.csv'] + (['map.pgm', 'map.yaml'] if role == 'mapping' else [])
            for name in names:
                if (directory/name).read_bytes() != (comparison/label/product/name).read_bytes():
                    raise ValueError(f'Standard control changed: {label}/{product}/{name}')
    (out/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {}
    for mode in ('first', 'centroid', 'all'):
        base = args.output/mode
        results = {}
        for label in ('direct', 'converted'):
            directory = base/label
            subprocess.run([sys.executable, '-m', 'tools.intel_voxel_ablation', args.data, args.comparison,
                            directory, '--trial', mode, label], check=True, timeout=120)
            results[label] = json.loads((directory/'results.json').read_bytes())
        results['between'] = {name: trajectory(base/'direct'/name/'trajectory.csv',
                                               base/'converted'/name/'trajectory.csv')
                              for name in ('map', 'known')}
        results['same_map'] = all((base/'direct/map'/name).read_bytes() == (base/'converted/map'/name).read_bytes()
                                  for name in ('map.yaml', 'map.pgm'))
        report[mode] = results
        print(mode, json.dumps(results), flush=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'comparison', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--trial', nargs=2, metavar=('MODE', 'SOURCE'))
    options = parser.parse_args()
    if options.trial:
        mode, label = options.trial
        if mode not in ('first', 'centroid', 'all') or label not in ('direct', 'converted'):
            parser.error('--trial expects {first,centroid,all} and {direct,converted}')
        trial(options.data, options.comparison, options.output, *options.trial)
    else:
        run(options)
