"""Report internal mapping revisits from a bag and its estimated trajectory."""
import argparse
import csv
import json
from collections import Counter, deque
from itertools import zip_longest
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .bag import load_config, scans
from .loop import diagnose
from .slam import world


def ambiguous(cloud, previous, fitted, overlap):
    """Look for a comparable translated alignment along the scan's long axis."""
    aligned = world(cloud, fitted)
    _, axes = np.linalg.eigh(np.cov(previous.T))
    tree, reverse = cKDTree(previous), cKDTree(aligned)
    axis = axes[:, -1]
    old_pos, new_pos = previous @ axis, aligned @ axis
    for offset in (-3., -2., -1., 1., 2., 3.):
        delta = offset * axis
        # Compare the shared field of view, excluding the strip lost at either end.
        use_new = (old_pos.min() <= new_pos + offset) & (new_pos + offset <= old_pos.max())
        use_old = (new_pos.min() <= old_pos - offset) & (old_pos - offset <= new_pos.max())
        if min(use_new.mean(), use_old.mean()) < .70:
            continue
        score = min((tree.query(aligned[use_new] + delta)[0] < .16).mean(),
                    (reverse.query(previous[use_old] - delta)[0] < .16).mean())
        if score >= max(.70, overlap - .05):
            return True
    return False


def search(stream, rows, *, stride=5, history=64, budget=64):
    if stride < 1 or history < 3 or budget < 1:
        raise ValueError('Require positive stride and budget, history >= 3')
    saved = deque(maxlen=history)
    counts = Counter()
    events = []
    previous_t = None
    for index, (scan, row) in enumerate(zip_longest(stream, rows)):
        if scan is None or row is None:
            raise ValueError('Scan and trajectory lengths differ')
        t = float(row['t'])
        pose = np.array([float(row[k]) for k in ('x', 'y', 'yaw')])
        if (not np.isfinite([t, scan.t, *pose]).all() or abs(t - scan.t) > .001
                or previous_t is not None and t <= previous_t):
            raise ValueError('Expected corresponding scans and increasing finite trajectory timestamps')
        previous_t = t
        counts['scans'] += 1
        if index % stride:
            counts['skipped_stride'] += 1
            continue
        if row['status'] not in ('origin', 'matched') or len(scan.xy) < 12:
            counts['skipped_unconfirmed_or_sparse'] += 1
            continue
        counts['sampled'] += 1
        # Bound retained cloud size and per-pair nearest-neighbour queries.
        xy = scan.xy[np.linspace(0, len(scan.xy)-1, min(len(scan.xy), 512), dtype=int)]
        if not np.isfinite(xy).all():
            raise ValueError('Expected finite XY returns')
        poses = np.asarray([item[2] for item in saved] + [pose])
        nearby = []
        for j, (_, old_t, old_pose, _) in enumerate(saved):
            if t - old_t < 4.:
                counts['too_recent'] += 1
                continue
            separation = np.linalg.norm(old_pose[:2] - pose[:2])
            if separation > 1.:
                counts['too_far'] += 1
                continue
            if np.linalg.norm(poses[j:, :2] - old_pose[:2], axis=1).max() < 2.:
                counts['no_excursion'] += 1
                continue
            nearby.append((separation, j))
        for _, j in sorted(nearby):
            if counts['tested'] >= budget:
                counts['skipped_budget'] += 1
                continue
            old_i, old_t, _, old_xy = saved[j]
            counts['tested'] += 1
            diag = diagnose(old_xy, xy, poses[j:])
            if not diag['loop_supported']:
                counts[diag['reason']] += 1
                continue
            fitted = diag['measured_first_from_last']
            if ambiguous(xy, old_xy, fitted, diag['symmetric_overlap']):
                counts['ambiguous_translation'] += 1
                continue
            events.append(dict(first=old_i, second=index, t0=old_t, t1=t,
                               measured=diag['measured_first_from_last'],
                               overlap=diag['symmetric_overlap'], separation_m=diag['geometric_distance_m']))
        if len(saved) == history:
            counts['evicted'] += 1
        saved.append((index, t, pose, xy))
    return dict(counts=dict(counts), candidates=events, stride=stride, history=history, budget=budget)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('trajectory', type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--history', type=int, default=64)
    parser.add_argument('--budget', type=int, default=64)
    args = parser.parse_args()
    try:
        with args.trajectory.open(encoding='utf-8-sig', newline='') as source:
            report = search(scans(args.bag, load_config(args.config)), csv.DictReader(source),
                            stride=args.stride, history=args.history, budget=args.budget)
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(2, f'{exc}\n')
