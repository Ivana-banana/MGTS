"""Measure obstacle reports on office walls under three pose sources (issue 154)."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
from rosbags.typesys import Stores, get_typestore
from scipy.ndimage import binary_erosion

from mgts_nav import demo
from mgts_nav.bag import load_config, scans
from mgts_nav.grid import read_map
from mgts_nav.obstacles import matched_poses, scan_groups
from mgts_nav.slam import world

TARGET = np.array([1.25, 2.10])


def record(root, role, seed, short=False, added=False):
    cfg = json.loads(demo.PROFILE.read_bytes())
    # The demo mount's horizontal axes are aligned with the body.
    assert cfg['base_from_lidar_xyzw'] == [0, 0, 0, 1]

    def returns(pose, elapsed, profile, rng, walls, dropout):
        rot = demo.rotation2(pose[2])
        origin = pose[:2] + rot @ profile['base_from_lidar_xyz_m'][:2]
        rays = demo.RAYS @ rot.T
        edge = walls[:, 1] - walls[:, 0]
        offset = walls[:, 0] - origin
        denominator = demo.cross(rays[:, None], edge)
        valid = np.abs(denominator) > 1e-10
        with np.errstate(divide='ignore', invalid='ignore'):
            along = demo.cross(offset, edge) / denominator
            wall_fraction = demo.cross(offset, rays[:, None]) / denominator
        distance = np.where(valid & (along > 0) & (wall_fraction >= 0) & (wall_fraction <= 1),
                            along, np.inf).min(axis=1)
        if added and elapsed >= 2.0:
            projection = rays @ (TARGET - origin)
            discriminant = .30**2 - (np.sum((TARGET-origin)**2) - projection**2)
            near = projection - np.sqrt(np.maximum(discriminant, 0))
            distance = np.minimum(distance, np.where((discriminant >= 0) & (near > 0), near, np.inf))
        keep = rng.random(len(rays)) > dropout
        measured = distance + rng.normal(0., .008, len(rays))
        keep &= np.isfinite(measured) & (measured <= (2.5 if short else np.inf))
        return np.column_stack((measured[keep, None]*demo.RAYS[keep], np.full(keep.sum(), .35))).astype('<f4')

    with patch.object(demo, 'lidar_points', returns):
        demo.record(root, role, cfg, get_typestore(Stores.ROS2_HUMBLE), seed)


def cli(module, *args):
    subprocess.run([sys.executable, '-X', 'utf8', '-m', module, *map(str, args)], check=True,
                   stdout=subprocess.DEVNULL)


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def evaluate(case, map_path, added):
    bag = case/'inspection'
    actual = case/'localized/trajectory.csv'
    truth = read_rows(case/'inspection_truth.csv')
    initial = [float(truth[0][key]) for key in ('x', 'y', 'yaw')]
    initial[0] += .45
    cli('mgts_nav', 'localize', bag, '--config', demo.PROFILE, '--map', map_path,
        '--initial', *initial, '--out', actual.parent)
    statuses = read_rows(actual)
    assert [float(r['t']) for r in truth] == [float(r['t']) for r in statuses]
    for name, shift in [('truth', 0.), ('offset', .45)]:
        with (case/f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['t','x','y','yaw','status'])
            writer.writeheader()
            for row, loc in zip(truth, statuses):
                writer.writerow(dict(row, x=float(row['x'])+shift, status=loc['status']))
    stream = list(scans(bag, load_config(demo.PROFILE)))
    pixels, meta, _ = read_map(map_path)
    margin = max(1, int(np.ceil(.12/meta['resolution'])))
    free = binary_erosion((1-pixels/255.) < meta['free_thresh'], structure=np.ones((3,3), bool),
                          iterations=margin)
    result = {}
    for name, path in [('located', actual), ('truth', case/'truth.csv'), ('offset', case/'offset.csv')]:
        report = case/f'{name}.json'
        cli('mgts_nav.obstacles', bag, '--config', demo.PROFILE, '--map', map_path, '--trajectory', path,
            '--out', report)
        data = json.loads(report.read_bytes())
        poses = matched_poses(path)
        groups = sum(bool(scan_groups(world(s.xy, poses[s.t]), free, meta)) for s in stream if s.t in poses)
        errors = [np.sum((poses[float(r['t'])][:2] - [float(r['x']),float(r['y'])])**2)
                  for r in truth if float(r['t']) in poses]
        events = data['events']
        onset = demo.START_NS['inspection'] / 1e9 + 2.
        other = [e for e in events if not (added and e['confirmed_s'] >= onset and
                 np.linalg.norm(np.array([e['x_m'], e['y_m']])-TARGET) <= .45)]
        result[name] = dict(scans=data['scans'], matched=data['matched_pose_scans'], group_scans=groups,
                            rmse_xy_m=float(np.sqrt(np.mean(errors))) if errors else None,
                            target_events=len(events)-len(other), other_events=len(other),
                            other_confirmation_scans=len({e['confirmed_s'] for e in other}))
    return result


def experiment(root):
    root.mkdir(parents=True)
    record(root, 'mapping', 37)
    cli('mgts_nav', 'map', root/'mapping', '--config', demo.PROFILE, '--out', root/'map')
    results = {}
    for seed in (53, 97):
        for short in (False, True):
            for added in (False, True):
                label = f"{seed}-{'short' if short else 'full'}-{'object' if added else 'static'}"
                case = root/label
                case.mkdir()
                record(case, 'inspection', seed, short, added)
                results[label] = evaluate(case, root/'map/map.yaml', added)
                print(label, flush=True)
    # Replay wall-only returns with the actual poses obtained while the cylinder was present.
    for seed in (53, 97):
        for view in ('full', 'short'):
            label = f'{seed}-{view}-object'
            path = root/label/'wall-replay.json'
            cli('mgts_nav.obstacles', root/f'{seed}-{view}-static'/'inspection',
                '--config', demo.PROFILE, '--map', root/'map/map.yaml',
                '--trajectory', root/label/'localized/trajectory.csv', '--out', path)
            replay = json.loads(path.read_bytes())
            assert replay['matched_pose_scans'] == results[label]['located']['matched']
            results[label]['wall_replay'] = dict(events=len(replay['events']),
                confirmation_scans=len({e['confirmed_s'] for e in replay['events']}))
    (root/'report.json').write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    experiment(parser.parse_args().out)
