"""Compare confirmed zone visits to the synthetic office tracks from issue 47."""
import argparse
import json
from pathlib import Path

from mgts_nav.bag import load_config, scans
from mgts_nav.demo import PROFILE
from mgts_nav.grid import read_map
from mgts_nav.metrics import read_csv
from mgts_nav.slam import track
from mgts_nav.zones import load_zones, visits
from tools.evaluate_motion_disagreement import perturb


def timestamps(intervals, zone, rows):
    return {row['t'] for name, first, last, *_ in intervals if name == zone
            for row in rows if first <= row['t'] <= last}


def compare(directory, zones):
    data = directory / 'data'
    points = read_map(directory / 'map.yaml')[2]
    stream = list(scans(data / 'inspection', load_config(PROFILE)))
    start = json.loads((data / 'initial_pose.json').read_bytes())
    refs = read_csv(data / 'inspection_truth.csv')
    expected = []
    for scan, ref in zip(stream, refs, strict=True):
        if abs(scan.t - ref.t) > 1e-6:
            raise ValueError('Unaligned inspection and reference timestamps')
        expected.append(dict(t=scan.t, x=ref.x, y=ref.y, status='matched'))
    truth = visits(expected, zones, .25)
    for case in ('clean', 'range3'):
        estimates = track(perturb(stream, case), start, map_points=points)
        rows = [dict(t=p.t, x=p.pose[0], y=p.pose[1], status=p.status) for p in estimates]
        if [r['t'] for r in rows] != [r['t'] for r in expected]:
            raise ValueError('Incomplete estimate')
        observed = visits(rows, zones, .25)
        counts = {}
        for zone in zones:
            actual = timestamps(observed, zone, rows)
            target = timestamps(truth, zone, expected)
            counts[zone] = dict(correct=len(actual & target), extra=len(actual - target),
                                missed=len(target - actual))
        yield dict(scene=directory.name, case=case, truth=truth, visits=observed, counts=counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path, help='Output of evaluate_motion_disagreement --pr-office')
    args = parser.parse_args()
    zones = load_zones(Path('configs/demo-zones.json'))
    for seed in (17, 101, 202, 303, 404):
        for row in compare(args.experiment / f'office-{seed}', zones):
            print(json.dumps(row, allow_nan=False))
