"""Reproduce the preregistered angular-recovery controls from issue 78."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from mgts_nav.bag import load_config, scans
from mgts_nav.demo import PROFILE, generate
from mgts_nav.grid import Grid, read_map
from mgts_nav.metrics import read_csv, wrap
from mgts_nav.slam import predict, track
from tools.heap1_turn_recovery import AngularRecovery


class CountSearches(AngularRecovery):
    searches = 0

    def candidate(self, xy, prior):
        self.searches += 1
        return super().candidate(xy, prior)


def trials(points, stream, start):
    tree = cKDTree(points)
    helper = CountSearches(tree, start)
    events = []
    onset = None
    estimates = []
    for r in track(stream, start, map_points=points, recovery=helper):
        helper.last = r.pose
        estimates.append(r)
        if helper.confirmations and helper.confirmations[-1] == r.t:
            events.append((len(estimates)-1, r.pose, r.t-onset))
        if r.status == 'matched':
            onset = None
        elif onset is None:
            onset = r.t
    normal = list(track(stream, start, map_points=points))
    isolated = CountSearches(tree, start)
    isolated_events = []
    for i, scan in enumerate(stream[:4]):
        proposal = isolated.update(scan, False)
        isolated.last = predict(isolated.last, None, scan)
        if proposal is not None:
            isolated_events.append((i, proposal, scan.t-stream[0].t))
    return estimates, normal, [(events, helper.searches), (isolated_events, isolated.searches)]


def error(pose, ref):
    return float(np.linalg.norm(pose[:2]-ref[1:3])), float(abs(np.degrees(wrap(pose[2]-ref.yaw))))


def experiment(out):
    out.mkdir(parents=True)
    cfg = load_config(PROFILE)
    runs = []
    for seed in (101, 202):
        maps, streams, starts, truths = {}, {}, {}, {}
        for layout in ('office', 'corridor'):
            root = out/f'{layout}-{seed}'
            data = root/'data'
            generate(data, layout=layout, seed=seed)
            grid = Grid(.08)
            list(track(scans(data/'mapping', cfg), np.zeros(3), grid=grid))
            grid.save(root)
            maps[layout] = read_map(root/'map.yaml')[2]
            stream = list(scans(data/'inspection', dict(cfg, odom=None)))
            if layout == 'corridor':
                stream = [replace(s, xy=s.xy[np.linalg.norm(s.xy-s.origin, axis=1) <= 3.]) for s in stream]
            streams[layout] = stream
            starts[layout] = np.array(json.loads((data/'initial_pose.json').read_bytes()))
            truths[layout] = data/'inspection_truth.csv'
        for layout, target, dx in [('office','office',0.), ('office','office',1.5),
                                   ('corridor','corridor',0.), ('corridor','corridor',1.5),
                                   ('office','corridor',0.)]:
            start = starts[layout] + [dx, 0., np.deg2rad(30)]
            runs.append((seed, layout, target, dx, truths[layout],
                         trials(maps[target], streams[layout], start)))

    report = []
    for seed, layout, target, dx, path, (estimates, normal, protocols) in runs:
        truth = read_csv(path)
        if len(truth) != 100 or not np.allclose([p.t for p in truth], [r.t for r in estimates],
                                                atol=1e-6, rtol=0):
            raise ValueError('Unexpected inspection timestamps')
        row = dict(seed=seed, layout=layout, map=target, dx=dx)
        confirmed_ids = {i for i, _, _ in protocols[0][0]}
        for key, route, exclude in [('baseline', normal, set()), ('experimental_ordinary', estimates, confirmed_ids)]:
            matches = [error(r.pose, truth[i]) for i,r in enumerate(route)
                       if i not in exclude and r.status == 'matched']
            row[key] = dict(matched=len(matches), false=sum(target != layout or m>.25 or a>10
                                                            for m,a in matches))
        for name, (events, searches) in zip(('track','isolated'), protocols, strict=True):
            details = []
            for i, pose, delay in events:
                xy, yaw = error(pose, truth[i])
                bx, ba = error(normal[i].pose, truth[i])
                details.append(dict(index=i, delay_s=delay, xy_m=xy, yaw_deg=yaw,
                                    correct=target == layout and xy <= .25 and yaw <= 10,
                                    baseline_correct=target == layout and normal[i].status == 'matched'
                                    and bx <= .25 and ba <= 10))
            row[name] = dict(searches=searches, confirmations=details, outcome='confirmed' if events
                             else 'refused' if searches else 'not_triggered')
        report.append(row)
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    experiment(parser.parse_args().out)
