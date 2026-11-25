"""Replay PR 43's map-point changes on held-out Velodyne scans with known starts.

Modes: ``centres`` (main), ``surface`` (first PR version, 3x3 mean of all map
points) and ``projection`` (``track(surface=True)``, the gated wall projection).
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from mgts_nav.bag import load_config, scans
from mgts_nav.grid import read_map
from mgts_nav.metrics import evaluate, Pose, read_csv
from mgts_nav.slam import track


# Kept identical to the proposal at d1209824e1868a9ff840ec621f616cb9ab4058e5.
def surface_points(points):
    """Replace grid cell centres by the mean of their occupied 3x3 neighbourhood.

    Cell centres of an occupancy grid describe a wall as a staircase that is
    offset by up to half a cell. The neighbourhood mean puts each point on the
    local wall line, so ICP is not biased toward one row of cells. The cell
    size is the median spacing between neighbouring map points.
    """
    points = np.asarray(points, dtype=float)
    tree = cKDTree(points)
    spacing = np.median(tree.query(points, k=2)[0][:, 1])
    if not np.isfinite(spacing) or spacing <= 0:
        return points
    neighbours = tree.query_ball_point(points, 1.5 * spacing)
    return np.array([points[ids].mean(axis=0) for ids in neighbours])


def compare_window(window, truth, points):
    if len(window) != 60 or len(truth) != 60 or not np.allclose([s.t for s in window], [r.t for r in truth],
                                                               atol=1e-6, rtol=0):
        raise ValueError('Expected 60 paired timestamps per window')
    window[0] = replace(window[0], gyro_delta=None)
    initial = np.array(truth[0][1:])
    modes = ('centres', 'surface', 'projection')
    times = {mode: [] for mode in modes}
    result = {}
    for i in range(3):
        for mode in (modes if i % 2 == 0 else modes[::-1]):
            tick = perf_counter()
            target = surface_points(points) if mode == 'surface' else points
            rows = list(track(window, initial, map_points=target, surface=mode == 'projection'))
            times[mode].append(perf_counter()-tick)
            metrics = evaluate([Pose(r.t, *r.pose) for r in rows], truth)
            residual = np.array([r.pose[:2] for r in rows]) - np.array([r[1:3] for r in truth])
            result[mode] = dict(matched=sum(r.status == 'matched' for r in rows),
                                ate_m=metrics['ate_xy_rmse_m'], rpe_m=metrics['rpe_translation_rmse_m'],
                                rpe_yaw_deg=np.degrees(metrics['rpe_rotation_rmse_rad']),
                                mean_xy_error_m=residual.mean(axis=0).tolist())
    for mode in result:
        result[mode]['seconds'] = times[mode]
        result[mode]['median_s'] = median(times[mode])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'holdout', 'map'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    _, _, points = read_map(args.map)
    report = {}
    for dataset, first, count in ((args.original, 2560, 60), (args.holdout, 2620, 180)):
        window = list(scans(dataset/'inspection', load_config(dataset/'profile.json')))
        truth = read_csv(dataset/'inspection-reference.csv')
        if len(window) != count or len(truth) != count:
            parser.error(f'Expected {count} scans in {dataset}')
        for offset in range(0, count, 60):
            report[first+offset] = compare_window(window[offset:offset+60], truth[offset:offset+60], points)
    print(json.dumps(report, indent=2, allow_nan=False))
