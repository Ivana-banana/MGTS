"""Grade every 3D fit of the official inspection and check the grades on revisits (#214).

The loop mirrors mgts_nav.localize3d: the same deskewed scans, every fourth point
held out of the fit, the same correction filter.  Each accepted fit is graded by
mgts_nav.match_quality.  Revisit pairs come from tools.evaluate_go2_localization
(revisits.json); a pair is "bad" when the 3D track is off the scan-to-scan
reference by more than 0.30 m or 5 degrees.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from mgts_nav.localize3d import CorrectionFilter, SurfaceMap, scans
from mgts_nav.match_quality import assess


def grade(inspection, map_npz):
    with np.load(map_npz) as data:
        surface = SurfaceMap(data['xyz'])
    filt = CorrectionFilter()
    rows = []
    for t, cloud, prior, _ in scans(inspection):
        held = np.arange(len(cloud)) % 4 == 0
        fit, accepted = surface.align(cloud[~held], prior)
        filt.update(t, prior, fit, accepted)
        q = assess(surface, cloud[~held], cloud[held], prior, fit, accepted)
        rows.append({'t': t, 'level': q.level, 'reasons': '+'.join(q.reasons), 'inliers': q.inliers,
                     'residual_prior_m': q.residual_prior_m, 'residual_fit_m': q.residual_fit_m,
                     'eigen_ratio': q.eigen_ratio, 'weakest_axis': q.weakest_axis,
                     'shift_m': q.shift_m, 'turn_rad': q.turn_rad})
    return rows


def pair_check(rows, revisits):
    out = {'bad': Counter(), 'good': Counter()}
    for r in revisits:
        if r['reference'] is None or '3d' not in r:
            continue
        bad = r['3d']['error_m'] > 0.30 or r['3d']['error_deg'] > 5
        levels = {rows[r['i']]['level'], rows[r['j']]['level']}
        out['bad' if bad else 'good']['pairs'] += 1
        out['bad' if bad else 'good']['with_weak_or_prediction'] += bool(levels - {'confident'})
    return {k: dict(v) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inspection', type=Path)
    parser.add_argument('map', type=Path, help='map.npz of localize3d')
    parser.add_argument('--revisits', type=Path, help='revisits.json of tools.evaluate_go2_localization')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = grade(args.inspection, args.map)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'quality.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {'scans': len(rows), 'levels': dict(Counter(r['level'] for r in rows)),
              'reasons': dict(Counter(x for r in rows for x in r['reasons'].split('+') if x)),
              'weakest_axis_when_degenerate': dict(Counter(r['weakest_axis'] for r in rows
                                                            if 'degenerate' in r['reasons']))}
    for key in ('inliers', 'eigen_ratio', 'shift_m'):
        values = np.array([r[key] for r in rows if r['level'] != 'prediction_only'])
        report[f'{key}_p5_p50_p95'] = np.round(np.percentile(values, [5, 50, 95]), 4).tolist()
    if args.revisits:
        report['revisit_pairs'] = pair_check(rows, json.loads(args.revisits.read_text(encoding='utf-8')))
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
