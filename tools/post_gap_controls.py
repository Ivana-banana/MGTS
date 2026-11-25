"""Run preregistered post-gap start controls on foreign, repeated and field maps."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from mgts_nav.bag import load_config, scans
from mgts_nav.demo import PROFILE, generate
from mgts_nav.grid import read_map
from mgts_nav.metrics import Pose, evaluate, read_csv, wrap
from tools.delayed_after_gap import search
from tools.evaluate_lidar_extrinsics import run


def corridor_maps(source, root):
    pixels, meta, _ = read_map(source)
    cells = round(20 / meta['resolution'])
    if cells < pixels.shape[1] or not np.isclose(cells * meta['resolution'], 20):
        raise ValueError('Copies must be disjoint and aligned to the grid')
    double = np.full((len(pixels), cells + pixels.shape[1]), 205, dtype=np.uint8)
    double[:, :pixels.shape[1]] = pixels
    double[:, cells:] = pixels
    shifted = dict(meta, origin=[meta['origin'][0]+20, *meta['origin'][1:]])
    for name, image, metadata in [('double', double, meta), ('shifted', pixels, shifted)]:
        path = root/name
        path.mkdir()
        Image.fromarray(image).save(path/'map.pgm')
        (path/'map.yaml').write_text(yaml.safe_dump(dict(metadata, image='map.pgm')), encoding='utf8')


def trial(data, config, map_path, method, reference, foreign):
    stream = list(scans(data/'inspection', dict(config, odom=None)))
    supplied = stream[:2] + stream[8:]
    report, tracked = search(supplied, *read_map(map_path), method)
    rows = list(tracked)
    assert report['uncovered_t'] + [r.t for r in rows] == [s.t for s in supplied]
    report.update(input_scans=len(supplied), source_scans=len(stream), covered=len(rows),
                  uncovered_fraction=len(report['uncovered_t'])/len(supplied))
    if not rows:
        return report
    if foreign:
        report['outcome'] = 'false_accept'
        return report
    truth = read_csv(reference)
    if len(truth) != len(stream) or not np.allclose([p.t for p in truth], [s.t for s in stream],
                                                    rtol=0, atol=1e-6):
        raise ValueError('Reference clock differs from lidar')
    index = min(range(len(truth)), key=lambda i: abs(truth[i].t-report['start_t']))
    ref = truth[index]
    pose = report['start']
    xy = float(np.hypot(pose[0]-ref.x, pose[1]-ref.y))
    yaw = abs(np.degrees(wrap(pose[2]-ref.yaw)))
    # Reference CSVs may round lidar timestamps to microseconds.
    truth_at = {s.t: Pose(s.t, *p[1:]) for s, p in zip(stream, truth)}
    estimated = [Pose(r.t, *r.pose) for r in rows]
    metrics = evaluate(estimated, [truth_at[r.t] for r in rows])
    report.update(start_index_in_source=index, start_error_m=xy, start_error_deg=yaw,
                  matched=sum(r.status == 'matched' for r in rows), ate_m=metrics['ate_xy_rmse_m'],
                  outcome='correct' if xy <= .25 and yaw <= 10 else 'false_accept')
    return report


def experiment(field, root):
    root.mkdir(parents=True)
    office, corridor = root/'office', root/'corridor'
    generate(office, layout='office', seed=29)
    generate(corridor, layout='corridor', seed=61)
    base = root/'corridor-map'
    run('map', corridor/'mapping', '--config', PROFILE, '--out', base)
    corridor_maps(base/'map.yaml', root)
    field_map = root/'field-map'
    run('map', field/'mapping', '--config', field/'profile.json', '--out', field_map)
    cases = [('foreign', office, PROFILE, base, office/'inspection_truth.csv'),
             ('double', corridor, PROFILE, root/'double', corridor/'inspection_truth.csv'),
             ('shifted', corridor, PROFILE, root/'shifted', corridor/'inspection_truth.csv'),
             ('heap', field, field/'profile.json', field_map, field/'inspection-reference.csv')]
    reports = {}
    for case, data, config, map_dir, reference in cases:
        for method in ('pairwise', 'keyframes'):
            report = trial(data, load_config(config), map_dir/'map.yaml', method, reference, case=='foreign')
            reports[f'{case}-{method}'] = report
            print(case, method, report['outcome'], flush=True)
    (root/'report.json').write_text(json.dumps(reports, indent=2, allow_nan=False), encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('field', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    experiment(args.field, args.output)
