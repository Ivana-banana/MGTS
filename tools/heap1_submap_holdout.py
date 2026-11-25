"""Replay the three fixed HEAP-1 holdout windows against the eight survey sections."""
import argparse
from itertools import islice
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mgts_nav.bag import load_config, scans
from mgts_nav.delayed_start import estimate_prepared, prepare_window
from mgts_nav.grid import read_map
from mgts_nav.metrics import read_csv, wrap


def compare(data, sections, output):
    profile = load_config(data/'profile.json')
    stream = iter(scans(data/'inspection', profile))
    reports = []
    for offset in range(0, 180, 60):
        samples = list(islice(stream, 60))
        if len(samples) != 60:
            raise ValueError('Expected three complete 60-scan windows')
        prepared = prepare_window(samples)
        results, starts = [], []
        for first in range(0, 300, 40):
            pixels, meta, _ = read_map(sections/f'section-{first}'/'map.yaml')
            stats = {}
            row = dict(first_survey_scan=first)
            try:
                start = estimate_prepared(prepared, pixels, meta, stats)
            except ValueError as exc:
                row['reason'] = str(exc)
            else:
                row['start'] = start.tolist()
                starts.append(row['start'])
            row['anchors'] = stats['anchors']
            results.append(row)
        reports.append(dict(first_frame=2620+offset, first_t=samples[0].t, sections=results,
                            accepted_sections=len(starts), start=starts[0] if len(starts) == 1 else None))
    if next(stream, None) is not None:
        raise ValueError('Expected exactly 180 inspection scans')
    (output/'blind.json').write_text(json.dumps(reports, indent=2, allow_nan=False), encoding='utf-8')
    reference = read_csv(data/'inspection-reference.csv')
    if len(reference) != 180:
        raise ValueError('Expected 180 reference poses')
    for row in reports:
        truth = reference[row['first_frame']-2620]
        if abs(truth.t-row['first_t']) > 1e-6:
            raise ValueError('Reference clock does not match inspection')
        if row['start'] is not None:
            x, y, yaw = row['start']
            row['error_xy_m'] = float(np.hypot(x-truth.x, y-truth.y))
            row['error_yaw_deg'] = float(np.rad2deg(abs(wrap(yaw-truth.yaw))))
    (output/'scored.json').write_text(json.dumps(reports, indent=2, allow_nan=False), encoding='utf-8')
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('data', 'sections', 'output'):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        stage = Path(tmp)/'result'
        stage.mkdir()
        reports = compare(args.data, args.sections, stage)
        stage.rename(args.output)
    for row in reports:
        print(row['first_frame'], row['accepted_sections'], row['start'])
