"""Compare occupancy confirmation across viewpoints on paired synthetic and field scans."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial import cKDTree

from mgts_nav.bag import Scan, load_config, scans
from mgts_nav.demo import WALLS, RAYS, cross, rotation2, pose_at
from mgts_nav.grid import Grid, read_map
from mgts_nav.map_eval import segment_distances
from mgts_nav.metrics import read_csv
from mgts_nav.slam import track, world

CELL = .08
DOOR = np.array([[2., -.8], [2., .35]])
THIN = np.array([[-2., 1.7], [-.6, 1.7]])
GLASS = np.array([[.2, -.8], [1.3, -.8]])


class StableGrid(Grid):
    def __init__(self):
        super().__init__(CELL)
        self.first = {}
        self.confirmed = set()

    def observe_at(self, t, origin, ends):
        super().observe(origin, ends)
        for key in map(tuple, np.unique(np.floor(ends / CELL).astype(int), axis=0)):
            if key in self.confirmed:
                continue
            if key not in self.first:
                self.first[key] = t, origin.copy()
            first_t, first_origin = self.first[key]
            if t - first_t >= 1 and np.linalg.norm(origin - first_origin) >= .35:
                self.confirmed.add(key)

    def finish(self):
        for key in self.first.keys() - self.confirmed:
            if self.evidence[key] > 0:
                self.evidence[key] = 0


def scene(run):
    result, truth = [], []
    for i in range(200 if run == 'mapping' else 100):
        t = i / 10
        pose = pose_at(run, t)
        truth.append(pose)
        segments = list(WALLS) + [THIN, GLASS]
        if run == 'mapping' and i < 80:
            segments.append(DOOR)
        if run == 'mapping' and 50 <= i < 150:
            person = np.array([-.7 + .25 * max(0, t-12), .05])
            segments.append(person + np.array([[0, -.2], [0, .2]]))
        rays = RAYS @ rotation2(pose[2]).T
        segments = np.asarray(segments)
        edge = segments[:, 1] - segments[:, 0]
        offset = segments[:, 0] - pose[:2]
        denom = cross(rays[:, None], edge)
        with np.errstate(divide='ignore', invalid='ignore'):
            along = cross(offset, edge) / denom
            fraction = cross(offset, rays[:, None]) / denom
        hits = np.where((along > .15) & (fraction >= 0) & (fraction <= 1), along, np.inf)
        # Sparse transmission through glass, with physical returns from walls behind it.
        hits[(np.arange(len(rays)) + i) % 10 != 0, len(WALLS)+1] = np.inf
        ranges = hits.min(axis=1)
        valid = np.isfinite(ranges)
        result.append(Scan(t, RAYS[valid] * ranges[valid, None], np.zeros(2), None, None))
    return result, np.array(truth)


def samples(lines):
    return np.concatenate([a + (b-a)*np.linspace(.01,.99,max(2,round(np.linalg.norm(b-a)/.04)))[:,None]
                           for a,b in np.asarray(lines)])


def compare(mapping, survey, inspection, truth, synthetic):
    baseline, stable = Grid(CELL), StableGrid()
    snapshot = {}
    for scan, pose in zip(mapping, survey, strict=True):
        origin, ends = world(scan.origin, pose), world(scan.xy, pose)
        baseline.observe(origin, ends)
        stable.observe_at(scan.t, origin, ends)
        if synthetic and scan.t == 7.9:
            interim = deepcopy(stable)
            interim.finish()
            for name, grid in [('baseline',baseline),('confirmed',interim)]:
                occupied = (np.array([k for k,v in grid.evidence.items() if v>=5])+.5)*CELL
                tree = cKDTree(occupied)
                person = [[[-.7,-.15],[-.7,.25]]]
                snapshot[name] = dict(closed_door_recall=float(np.mean(tree.query(samples([DOOR]))[0]<=.12)),
                                      person_trace=float(np.mean(tree.query(samples(person))[0]<=.12)))
    stable.finish()
    result = {}
    for name, grid in [('baseline', baseline), ('confirmed', stable)]:
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            grid.save(directory)
            points = read_map(directory/'map.yaml')[2]
        tree = cKDTree(points)
        estimates = list(track(inspection, truth[0], map_points=points))
        distances = [tree.query(world(s.xy, p))[0] for s,p in zip(inspection, truth, strict=True)]
        row = dict(occupied=len(points), matched=sum(e.status=='matched' for e in estimates),
                   scans=len(estimates), revisit_support=float(np.mean(np.concatenate(distances)<.16)),
                   ate_m=float(np.sqrt(np.mean(np.sum((np.array([e.pose[:2] for e in estimates])-truth[:,:2])**2, axis=1)))))
        if synthetic:
            row.update(snapshot[name])
            lines = np.concatenate((WALLS,[GLASS,THIN]))
            row['false_occupied'] = int(np.sum(segment_distances(points,lines)>.12))
            for label, segments in [('all',lines),('glass',[GLASS]),('thin',[THIN])]:
                target = samples(segments)
                row[label+'_recall'] = float(np.mean(tree.query(target)[0] <= .12))
                row[label+'_false_free'] = float(np.mean([grid.evidence.get(tuple(c),0)<0
                    for c in np.floor(target/CELL).astype(int)]))
            row['open_door_residual_fraction'] = float(np.mean(tree.query(samples([DOOR]))[0]<=.12))
        result[name] = row
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--field', type=Path)
    parser.add_argument('--survey-trajectory', type=Path)
    args = parser.parse_args()
    if args.field:
        cfg = load_config(args.field/'profile.json')
        mapping, inspection = [list(scans(args.field/name,cfg)) for name in ('mapping','inspection')]
        survey, truth = [read_csv(p) for p in (args.survey_trajectory,args.field/'inspection-reference.csv')]
        for stream, poses in ((mapping,survey),(inspection,truth)):
            if any(abs(s.t-p.t)>1e-6 for s,p in zip(stream, poses, strict=True)):
                raise ValueError('Pose/scan time mismatch')
        survey, truth = [np.array([p[1:] for p in rows]) for rows in (survey,truth)]
    else:
        mapping, survey = scene('mapping')
        inspection, truth = scene('inspection')
    print(json.dumps(compare(mapping,survey,inspection,truth,not args.field),indent=2))
