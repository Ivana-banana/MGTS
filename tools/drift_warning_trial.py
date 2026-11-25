"""Evaluate pre-registered scan-level warnings on known-start inspection outputs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from mgts_nav.bag import load_config, scans
from mgts_nav.grid import read_map
from mgts_nav.metrics import cadence_limit, interpolate, read_csv
from mgts_nav.observability import diagnose
from mgts_nav.slam import relative, world
from tools.heap1_warning_score import counts, record


def predict(scan, previous, pose, previous_pose, support, tree, odom_mode):
    try:
        geometry = diagnose(scan.xy)['normal_eigenvalues'][0]
    except ValueError:
        geometry = None
    distances, _ = tree.query(world(scan.xy, pose))
    disagreement = False
    if previous is not None and 0 < scan.t - previous.t <= .25:
        delta = relative(previous_pose, pose)
        if scan.gyro_delta is not None:
            angle = delta[2] - scan.gyro_delta
            disagreement = abs(np.arctan2(np.sin(angle), np.cos(angle))) > np.deg2rad(3)
        if odom_mode != 'absent' and previous.odom is not None and scan.odom is not None:
            odom = relative(previous.odom, scan.odom)
            if odom_mode == 'drift':
                odom[0] += .08 * (scan.t - previous.t)
            disagreement |= np.linalg.norm(delta[:2] - odom[:2]) > .05
    votes = [support < .55, np.quantile(distances, .75) > .25,
             geometry is not None and geometry < .12, disagreement]
    return bool(votes[0]), sum(map(int, votes)) >= 2


def evaluate(rows, truth, index):
    times = [p.t for p in truth]
    gap = cadence_limit(truth)
    tally = counts()
    episodes = []
    start = previous = None
    for t, pose, predictions in rows:
        target = interpolate(truth, times, t, gap)
        if target is None or predictions is None:
            start = None
            continue
        large = np.linalg.norm(pose[:2] - target[1:3]) > .20
        alert = predictions[index]
        record(tally, alert, large)
        if large:
            if start is None or t - previous > .25:
                start = t
                episodes.append(None)
            if alert and episodes[-1] is None:
                episodes[-1] = t - start
        else:
            start = None
        previous = t
    elapsed = rows[-1][0] - rows[0][0]
    return dict(**tally, coverage=sum(tally.values()) / len(rows),
                first_signal_seconds=episodes, missed_episodes=episodes.count(None),
                false_signals_per_min=tally['fp'] * 60 / elapsed if elapsed else None)


def run(args):
    with args.trajectory.open(encoding='utf-8-sig', newline='') as stream:
        trajectory = list(csv.DictReader(stream))
    poses = read_csv(args.trajectory)
    tree = cKDTree(read_map(args.map)[2])
    by_time = {scan.t: scan for scan in scans(args.bag, load_config(args.profile))}
    result = []
    previous = previous_pose = None
    odom_scans = 0
    for row, entry in zip(trajectory, poses, strict=True):
        scan = by_time[entry.t]
        pose = np.asarray(entry[1:])
        predictions = None
        odom_scans += scan.odom is not None
        if row['status'] == 'matched' and row['inlier_fraction']:
            predictions = predict(scan, previous, pose, previous_pose,
                                  float(row['inlier_fraction']), tree, args.odom)
        result.append((scan.t, pose, predictions))
        previous, previous_pose = scan, pose
    truth = read_csv(args.truth)
    return dict(odom_scans=odom_scans, scans=len(result),
                baseline=evaluate(result, truth, 0), combined=evaluate(result, truth, 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bag', 'profile', 'map', 'trajectory', 'truth'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--odom', choices=('recorded', 'absent', 'drift'), default='recorded')
    print(json.dumps(run(parser.parse_args()), indent=2, allow_nan=False))
