"""Reproduce the fixed motion-disagreement controls for issue 47."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from mgts_nav.bag import load_config, scans
from mgts_nav.demo import PROFILE, generate
from mgts_nav.grid import Grid, read_map
from mgts_nav.metrics import read_csv, wrap
from mgts_nav.motion_disagreement import windows
from mgts_nav.slam import track


CASES = ('clean', 'range3', 'drift', 'reset', 'no_odom', 'range3_no_odom')


def perturb(stream, case):
    for scan in stream:
        xy, odom = scan.xy, scan.odom
        elapsed = scan.t - stream[0].t
        if case.startswith('range3'):
            xy = xy[np.linalg.norm(xy, axis=1) <= 3.]
        if 'no_odom' in case:
            odom = None
        elif case == 'drift':
            odom = odom + [max(0., elapsed - 3.) * .30, 0., 0.]
        elif case == 'reset' and elapsed >= 5.:
            odom = odom + [1., 0., 0.]
        yield replace(scan, xy=xy, odom=odom)


def score(observations, estimates, truth):
    tp = fp = fn = tn = unobserved = 0
    episodes = []
    onset = None
    errors = []
    for obs, pose, ref in zip(observations, estimates, truth, strict=True):
        if abs(pose.t - ref.t) > 1e-6:
            raise ValueError('Unaligned truth')
        error = float(np.linalg.norm(pose.pose[:2] - ref[1:3]))
        errors.append(error)
        bad = error > .25 or abs(np.degrees(wrap(pose.pose[2] - ref.yaw))) > 10
        if bad and onset is None:
            onset = pose.t
            episodes.append(dict(onset_s=onset, delay_s=None))
        elif not bad:
            onset = None
        alert = obs['alert']
        if alert is None:
            unobserved += bad
        elif alert and bad:
            tp += 1
            if episodes[-1]['delay_s'] is None:
                episodes[-1]['delay_s'] = round(pose.t - onset, 3)
        elif alert:
            fp += 1
        elif bad:
            fn += 1
        else:
            tn += 1
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, unobserved_bad=int(unobserved), episodes=episodes,
                matched=sum(p.status == 'matched' for p in estimates),
                ate_m=float(np.sqrt(np.mean(np.square(errors)))),
                missing_translation=sum(o['translation_m'] is None for o in observations),
                missing_yaw=sum(o['yaw_deg'] is None for o in observations))


def experiment(root, pr_office=False):
    root.mkdir(parents=True)
    cfg = load_config(PROFILE)
    report = []
    groups = {'office': (17, 101, 202, 303, 404)} if pr_office else dict(office=(41, 73), corridor=(41, 73))
    for layout, seeds in groups.items():
        for seed in seeds:
            work = root / f'{layout}-{seed}'
            data = work / 'data'
            generate(data, layout=layout, seed=seed)
            grid = Grid(.08)
            list(track(scans(data/'mapping', cfg), np.zeros(3), grid=grid))
            grid.save(work)
            points = read_map(work/'map.yaml')[2]
            stream = list(scans(data/'inspection', cfg))
            start = json.loads((data/'initial_pose.json').read_bytes())
            for case in (('clean', 'range3') if pr_office else CASES):
                modified = list(perturb(stream, case))
                poses = list(track(modified, start, map_points=points))
                obs = list(windows(modified, poses))
                result = score(obs, poses, read_csv(data/'inspection_truth.csv'))
                report.append(dict(layout=layout, seed=seed, case=case, **result))
                (work/f'{case}-windows.json').write_text(json.dumps(obs, allow_nan=False))
    (root/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--pr-office', action='store_true', help='Repeat clean and 3 m on the five PR 43 office seeds')
    args = parser.parse_args()
    experiment(args.out, args.pr_office)
