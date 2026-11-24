"""Score the three preselected HEAP-1 inspection windows against CPT7 after blind localization."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from mgts_nav.bag import load_config, scans
from mgts_nav.delayed_start import estimate_start
from mgts_nav.grid import read_map
from mgts_nav.metrics import Pose, evaluate, read_csv, wrap
from mgts_nav.slam import track


def score(rows, truth):
    return dict(matched=sum(r.status == 'matched' for r in rows),
                ate_m=evaluate([Pose(r.t, *r.pose) for r in rows], truth)['ate_xy_rmse_m'])


def compare(data, map_path):
    pixels, meta, points = read_map(map_path)
    inspection = list(scans(data / 'inspection', load_config(data / 'profile.json')))
    if len(inspection) != 180 or abs(inspection[0].t - 1731588674.081123) > 1e-6:
        raise ValueError('Expected exactly 180 scans from heap1_revisit.py --held-out')
    trials = []
    for offset in range(0, 180, 60):
        window = inspection[offset:offset+60]
        # A standalone window has no preceding IMU span at its first timestamp.
        window[0] = replace(window[0], gyro_delta=None)
        result = dict(first_frame=2620+offset)
        try:
            start = estimate_start(window, pixels, meta, result)
        except ValueError as exc:
            result.update(outcome='refused', reason=str(exc))
            rows = []
            start = None
        else:
            rows = list(track(window, start, map_points=points))
        trials.append((result, window, start, rows))

    reference = read_csv(data / 'inspection-reference.csv')
    map_track = cKDTree([row[1:3] for row in read_csv(data / 'mapping-reference.csv')])
    if len(reference) != len(inspection) or not np.allclose([r.t for r in reference], [s.t for s in inspection],
                                                           atol=1e-6, rtol=0):
        raise ValueError('CPT7 reference must cover the same timestamps')
    for index, (result, window, start, rows) in enumerate(trials):
        truth = reference[60*index:60*(index+1)]
        anchor = np.array(truth[0][1:])
        positions = np.array([p[1:3] for p in truth])
        result['max_displacement_from_window_start_m'] = float(np.linalg.norm(positions-anchor[:2], axis=1).max())
        result['yaw_range_deg'] = float(np.degrees(np.ptp(np.unwrap([p.yaw for p in truth]))))
        if start is not None:
            xy = float(np.linalg.norm(start[:2]-anchor[:2]))
            angle = float(np.degrees(abs(wrap(start[2]-anchor[2]))))
            result.update(outcome='correct' if xy <= .25 and angle <= 10 else 'false_accept',
                          start_xy_m=xy, start_yaw_deg=angle, blind=score(rows, truth))
        result['known_start'] = score(list(track(window, anchor, map_points=points)), truth)
        distance = map_track.query(positions)[0]
        result['distance_to_mapping_route_m'] = [float(distance.min()), float(distance.max())]
    return [result for result, *_ in trials]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('map', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.data, args.map), indent=2, allow_nan=False))
