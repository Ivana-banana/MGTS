"""Compare existing recovery outputs with a bounded angular recovery experiment."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree

from mgts_nav.bag import load_config, scans
from mgts_nav.grid import read_map
from mgts_nav.initialization import estimate_initial
from mgts_nav.metrics import Pose, evaluate, read_csv, wrap
from mgts_nav.slam import match, predict, relative, track, world


def consistent(a, b):
    d = relative(a, b)
    return np.linalg.norm(d[:2]) <= .20 and abs(d[2]) <= np.deg2rad(6)


class AngularRecovery:
    def __init__(self, tree, start):
        self.tree = tree
        self.last = start
        self.misses = 0
        self.pending = None
        self.attempts = 0
        self.confirmations = []

    def candidate(self, xy, prior):
        fits = []
        for angle in range(-45, 46, 5):
            pose, support, ok = match(xy, self.tree, prior + [0., 0., np.deg2rad(angle)])
            if ok and support >= .55 and np.linalg.norm(pose[:2]-prior[:2]) <= .45:
                fits.append((support, pose))
        if not fits:
            return None
        support, best = max(fits, key=lambda fit: fit[0])
        if any(s >= support-.05 and not consistent(best, p) for s, p in fits):
            return None
        aligned = world(xy, best)
        axis = np.linalg.eigh(np.cov(aligned.T))[1][:, -1]
        for shift in (-3., -2., -1., 1., 2., 3.):
            alternate = self.tree.query(aligned + shift*axis)[0]
            if np.mean(alternate < .16) >= support-.05:
                return None
        return best

    def update(self, scan, matched):
        if matched:
            self.misses = 0
            self.pending = None
            return None
        self.misses += 1
        if self.pending is None and self.misses < 3:
            return None
        self.misses = 0
        prior = predict(self.last, None, scan)
        observed = self.candidate(scan.xy, prior)
        previous, self.pending = self.pending, None
        if observed is None:
            return None
        if previous is None:
            self.attempts += 1
            self.pending = (scan.t, observed)
        elif 0 < scan.t-previous[0] <= 1 and consistent(predict(previous[1], None, scan), observed):
            self.confirmations.append(scan.t)
            return observed
        return None


def evaluate_route(poses, statuses, truth, end, confirmed):
    if len(poses) != len(truth) or not np.allclose([p.t for p in poses], [p.t for p in truth], atol=1e-6, rtol=0):
        raise ValueError('Reference timestamps differ')
    metrics = evaluate(poses, truth)
    errors = [dict(t=p.t, xy_m=float(np.hypot(p.x-q.x, p.y-q.y)),
                   yaw_deg=abs(np.degrees(wrap(p.yaw-q.yaw)))) for p, q in zip(poses, truth)]
    lost = next((p.t for p, status in zip(poses, statuses) if status == 'prediction_only'), None)
    regained = next((p.t for p, status in zip(poses, statuses)
                     if lost is not None and p.t > lost and status == 'matched'), None)
    return dict(matched=statuses.count('matched'), ate_m=metrics['ate_xy_rmse_m'],
                yaw_rmse_deg=np.degrees(metrics['absolute_yaw_rmse_rad']),
                max_yaw_deg=max(e['yaw_deg'] for e in errors),
                delay_from_loss_s=None if regained is None else regained-lost,
                time_relative_to_gap_end_s=None if regained is None else regained-end,
                confirmation_errors=[dict(**e, after_gap_s=e['t']-end) for e in errors if e['t'] in confirmed])


def experiment(data, solution, previous, output):
    start = np.array(json.loads((solution/'inspection/summary.json').read_bytes())['initial_pose'])
    cfg = load_config(data/'profile.json')
    pixels, meta, points = read_map(solution/'map/map.yaml')
    original = json.loads((previous/'report.json').read_bytes())
    routes = []
    output.mkdir(parents=True)
    for label in ('control', 'half', 'one'):
        bag = data/'inspection' if label == 'control' else previous/f'bag-{label}'
        stock_dir = output/f'stock-{label}'
        subprocess.run([sys.executable, '-m', 'mgts_nav', 'localize', str(bag), '--config',
                        str(data/'profile.json'), '--map', str(solution/'map/map.yaml'),
                        '--initial', *map(str, start), '--recover', '--out', str(stock_dir)], check=True)
        input_scans = list(scans(bag, cfg))
        helper = AngularRecovery(cKDTree(points), start)
        poses, statuses = [], []
        for row in track(input_scans, start, map_points=points, recovery=helper):
            helper.last = row.pose
            poses.append(Pose(row.t, *row.pose))
            statuses.append(row.status)
        end = poses[0].t + original[label]['end_s']
        routes.append((label, 'angular', poses, statuses, end, helper.confirmations,
                       dict(proposals=helper.attempts, recoveries=len(helper.confirmations))))
        for method, directory in [('off', previous/label), ('stock', stock_dir)]:
            with (directory/'trajectory.csv').open(newline='') as stream:
                flags = [row['status'] for row in csv.DictReader(stream)]
            summary = json.loads((directory/'summary.json').read_bytes())
            stats = summary.get('recovery', {})
            recorded = read_csv(directory/'trajectory.csv')
            if stats.get('attempts', 0):
                first_try = next(i for i in range(2, len(flags))
                                 if flags[i-2:i+1] == ['prediction_only']*3)
                try:
                    estimate_initial(input_scans[first_try].xy, pixels, meta)
                except ValueError as exc:
                    stats['first_search_result'] = str(exc)
                else:
                    stats['first_search_result'] = 'candidate'
            confirmed = []
            if stats.get('recoveries', 0):
                # Recovery.update can return a pose only after ordinary ICP failed on that scan.
                prior = start
                for scan, pose, status in zip(input_scans, recorded, flags, strict=True):
                    if status == 'matched' and not match(scan.xy, helper.tree, predict(prior, None, scan))[2]:
                        confirmed.append(scan.t)
                    prior = np.array(pose[1:])
                if len(confirmed) != stats['recoveries']:
                    raise ValueError('Confirmation timestamps disagree with stock counters')
            routes.append((label, method, recorded, flags, end, confirmed, stats))
    truth = read_csv(data/'inspection-reference.csv')
    result = {f'{label}-{method}': dict(**evaluate_route(poses, statuses, truth, end, confirmed), **stats)
              for label, method, poses, statuses, end, confirmed, stats in routes}
    with (output/'report.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'solution', 'previous', 'output'):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    experiment(args.data, args.solution, args.previous, args.output)
