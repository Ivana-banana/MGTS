"""Evaluate a delayed global start using up to six scans and scan-to-scan motion."""
import argparse
import json
from dataclasses import dataclass, replace
from itertools import combinations, islice
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from .bag import Scan, load_config, scans
from .coarse_start import estimate
from .grid import Grid, read_map
from .metrics import Pose, evaluate, read_csv, wrap
from .slam import match, relative, track, world


START_SEARCHES = ('coarse', 'delayed', 'delayed-keyframes')
MAX_WINDOW_SCANS = 51


def compose(pose, step):
    return np.r_[world(step[:2], pose), wrap(pose[2] + step[2])]


@dataclass(frozen=True)
class PreparedWindow:
    rows: list[Scan]
    local: list[np.ndarray]
    rejection: str | None = None
    rejected_pair: dict | None = None


def _local_trajectory(rows, local_method):
    if local_method == 'keyframes':
        stripped = (replace(scan, odom=None, gyro_delta=scan.gyro_delta if i else None)
                    for i, scan in enumerate(rows))
        local = []
        for row in track(stripped, np.zeros(3), grid=Grid(.08)):
            if row.status == 'prediction_only':
                raise ValueError('Unsupported local keyframe motion')
            local.append(row.pose)
    elif local_method == 'pairwise':
        local = [np.zeros(3)]
        for previous, current in zip(rows, rows[1:]):
            seed = np.array([0., 0., current.gyro_delta or 0.])
            step, _, ok = match(current.xy, cKDTree(previous.xy), seed)
            if not ok:
                raise ValueError('Unsupported local scan-to-scan motion')
            local.append(compose(local[-1], step))
    else:
        raise ValueError(f'Unknown local method: {local_method}')
    return local


def prepare_window(window, *, local_method='pairwise'):
    """Compute motion once, including a repeatable rejection for an unusable window."""
    rows = list(islice(window, MAX_WINDOW_SCANS))
    if len(rows) < 21:
        return PreparedWindow(rows, [], 'Need at least three global search slots')
    for i, (previous, current) in enumerate(zip(rows, rows[1:])):
        dt = current.t - previous.t
        causes = []
        if not 0 < dt <= .35:
            causes.append('time_interval')
        if min(len(previous.xy), len(current.xy)) < 12:
            causes.append('too_few_points')
        if causes:
            pair = dict(indices=[i, i+1], times_s=[previous.t, current.t], interval_s=dt,
                        points=[len(previous.xy), len(current.xy)], causes=causes)
            return PreparedWindow(rows, [], 'Gap in the local scan chain', pair)
    try:
        local = _local_trajectory(rows, local_method)
    except ValueError as exc:
        return PreparedWindow(rows, [], str(exc))
    return PreparedWindow(rows, local)


def estimate_start(window, pixels, meta, stats, *, local_method='pairwise'):
    """Return pose at the first scan. No map-frame prior or kinematic odometry is used."""
    return estimate_prepared(prepare_window(window, local_method=local_method), pixels, meta, stats)


def estimate_prepared(prepared, pixels, meta, stats):
    """Match one map with an already evaluated local scan chain."""
    stats['anchors'] = []
    if prepared.rejection is not None:
        if prepared.rejected_pair is not None:
            stats['rejected_pair'] = prepared.rejected_pair
        raise ValueError(prepared.rejection)
    rows, local = prepared.rows, prepared.local
    stats['local_end'] = local[-1].tolist()

    starts, anchors = [], []
    for i in range(0, len(rows), 10):
        diag = {}
        slot = dict(index=i)
        try:
            anchor = estimate(rows[i], pixels, meta, diag)
        except ValueError as exc:
            slot['rejection'] = str(exc)
        else:
            # map <- scan_i <- scan_0.
            starts.append(compose(anchor, relative(local[i], np.zeros(3))))
            anchors.append(anchor)
        slot['score'] = diag.get('best_score')
        slot['candidate'] = diag.get('best_refined_pose')
        slot['rival_score'] = diag.get('rival_score')
        stats['anchors'].append(slot)
    if len(starts) < 3:
        raise ValueError('Fewer than three accepted global poses')
    if not any(np.linalg.norm(a[:2]-b[:2]) >= .25 or abs(wrap(a[2]-b[2])) >= np.deg2rad(10)
               for a, b in combinations(anchors, 2)):
        raise ValueError('Insufficient change of viewpoint')
    for a, b in combinations(starts, 2):
        if np.linalg.norm(a[:2]-b[:2]) > .20 or abs(wrap(a[2]-b[2])) > np.deg2rad(6):
            raise ValueError('Global poses disagree with local motion')
    x, y, angles = np.asarray(starts).T
    return np.array([np.median(x), np.median(y), np.arctan2(np.sin(angles).sum(), np.cos(angles).sum())])


def search_prefix(stream, method, pixels, meta, diagnostics):
    if method not in START_SEARCHES:
        raise ValueError(f'Unknown start search: {method}')
    prefix = list(islice(stream, 1 if method == 'coarse' else MAX_WINDOW_SCANS))
    if not prefix:
        raise ValueError('No lidar scans in recording')
    if method == 'coarse':
        start = estimate(prefix[0], pixels, meta, diagnostics)
    else:
        local_method = 'keyframes' if method == 'delayed-keyframes' else 'pairwise'
        start = estimate_start(prefix, pixels, meta, diagnostics, local_method=local_method)
    return start, prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bag', 'profile', 'map'):
        parser.add_argument(name)
    parser.add_argument('--reference', help='Evaluate a completed search against a held-out CSV')
    parser.add_argument('--local-method', choices=['pairwise', 'keyframes'], default='pairwise')
    parser.add_argument('--pose-out', type=Path, help='New JSON file usable as localize --initial-file')
    args = parser.parse_args()
    cfg = load_config(args.profile)
    pixels, meta, points = read_map(args.map)
    report = {}
    tick = perf_counter()
    try:
        start = estimate_start(scans(args.bag, cfg), pixels, meta, report, local_method=args.local_method)
    except ValueError as exc:
        report.update(outcome='refused', reason=str(exc), search_s=perf_counter() - tick)
        print(json.dumps(report, indent=2, allow_nan=False))
        parser.exit(3)
    report.update(outcome='accepted', start=start.tolist(), search_s=perf_counter() - tick)
    if args.reference:
        rows = list(track(scans(args.bag, cfg), start, map_points=points))
        # Reference data are reserved for evaluation after tracking.
        truth = read_csv(args.reference)
        if len(truth) != len(rows) or not np.allclose([p.t for p in truth], [r.t for r in rows],
                                                       atol=1e-6, rtol=0):
            raise ValueError('Reference timestamps must match the scan timestamps')
        delta = relative(np.array(truth[0][1:]), start)
        xy, angle = float(np.linalg.norm(delta[:2])), float(np.rad2deg(abs(delta[2])))
        report.update(outcome='correct' if xy <= .25 and angle <= 10 else 'false_accept',
                      start_error_m=xy, start_error_deg=angle, matched=sum(r.status == 'matched' for r in rows),
                      metrics=evaluate([Pose(r.t, *r.pose) for r in rows], truth))
    if args.pose_out:
        try:
            with args.pose_out.open('x', encoding='utf-8') as target:
                target.write(json.dumps(start.tolist(), allow_nan=False) + '\n')
        except OSError as exc:
            parser.exit(2, f'{exc}\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
