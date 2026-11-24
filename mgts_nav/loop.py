"""Diagnose a possible endpoint revisit without changing the map or trajectory."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .bag import load_config, scans
from .slam import match, relative, world


def diagnose(first_xy, last_xy, trajectory):
    """Test endpoint proximity and symmetric lidar overlap in the first base frame."""
    trajectory = np.asarray(trajectory, dtype=float)
    if (trajectory.ndim != 2 or trajectory.shape[1] != 3 or len(trajectory) < 2
            or not np.isfinite(trajectory).all()):
        raise ValueError('Expected at least two finite trajectory poses')
    prior = relative(trajectory[0], trajectory[-1])
    separation = float(np.linalg.norm(prior[:2]))
    excursion = float(np.linalg.norm(trajectory[:, :2] - trajectory[0, :2], axis=1).max())
    report = dict(endpoint_distance_m=separation, endpoint_yaw_rad=float(prior[2]),
                  max_excursion_m=excursion, geometric_distance_m=None,
                  symmetric_overlap=None, loop_supported=False)
    if excursion < 1.0:
        return dict(report, reason='no_excursion')
    for cloud in (first_xy, last_xy):
        if cloud.ndim != 2 or cloud.shape[1] != 2 or not np.isfinite(cloud).all():
            raise ValueError('Expected two finite XY scans')
        if len(cloud) < 12:
            return dict(report, reason='too_few_endpoints')
    # A good registration alone also happens for open paths in the same room.
    # Compare its inferred sensor separation separately from scan overlap.
    tree = cKDTree(first_xy)
    fitted, _, accepted = match(last_xy, tree, prior, radius=.65)
    if not accepted:
        return dict(report, reason='registration_failed')
    aligned = world(last_xy, fitted)
    forward = (tree.query(aligned)[0] < .16).mean()
    backward = (cKDTree(aligned).query(first_xy)[0] < .16).mean()
    overlap = float(min(forward, backward))
    fitted_distance = float(np.linalg.norm(fitted[:2]))
    supported = separation <= 1.0 and fitted_distance <= .45 and overlap >= .70
    return dict(report, geometric_distance_m=fitted_distance, symmetric_overlap=overlap,
                measured_first_from_last=fitted.tolist(),
                loop_supported=supported, reason='supported' if supported else 'outside_gates')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('trajectory', type=Path)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    try:
        with args.trajectory.open(encoding='utf-8-sig', newline='') as source:
            rows = list(csv.DictReader(source))
        try:
            poses = [[float(row[key]) for key in ('x', 'y', 'yaw')] for row in rows]
            times = [float(row['t']) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'{args.trajectory}: needs numeric columns t,x,y,yaw') from exc
        stream = iter(scans(args.bag, load_config(args.config)))
        first = next(stream, None)
        if first is None or len(rows) < 2:
            parser.error('Recording and trajectory need at least two poses')
        last = first
        for last in stream:
            pass
        if max(abs(times[0] - first.t), abs(times[-1] - last.t)) > .001:
            parser.error('Trajectory endpoints must correspond to the first and last lidar scans')
        print(json.dumps(diagnose(first.xy, last.xy, poses), indent=2, allow_nan=False))
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
