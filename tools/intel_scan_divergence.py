"""Find the first divergent Intel Lab scan and replay its ICP with a fixed history."""
import argparse
import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mgts_nav import slam
from mgts_nav.bag import cloud_xyz, load_config, reader, scans


class TrajectoryOnly:
    """track builds its KD-tree from keyframes; the output raster is write-only."""
    def observe(self, origin, endpoints):
        pass


def saved(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def raw_scans(path, indices):
    selected = {}
    with reader(path) as bag:
        lidar = [c for c in bag.connections if c.topic == '/front_laser']
        for i, (c, _, payload) in enumerate(bag.messages(connections=lidar)):
            if i in indices:
                selected[i] = cloud_xyz(bag.deserialize(payload, c.msgtype))[:, :2].astype(float)
    return selected


def representatives(xy, filtered):
    _, ids = np.unique(np.floor(xy / .06), axis=0, return_index=True)
    np.testing.assert_array_equal(xy[ids], filtered)
    return ids


def compare_matching(streams, ids, tree, seed):
    details, pairs, final_pairs = [], [], []
    for xy, beams in zip(streams, ids):
        distance, targets = tree.query(slam.world(xy, seed), distance_upper_bound=.45)
        valid = np.isfinite(distance)
        keep = valid & (distance <= max(.06, np.quantile(distance[valid], .85)))
        pairs.append(set(zip(beams[keep], targets[keep])))
        pose, support, ok = slam.match(xy, tree, seed)
        fitted = set()
        if ok:
            residual, target = tree.query(slam.world(xy, pose))
            inside = residual < .16
            fitted = set(zip(beams[inside], target[inside]))
        final_pairs.append(fitted)
        details.append(dict(trimmed_at_seed=int(keep.sum()), pose=pose.tolist(), accepted=ok,
                            support=support, accepted_pairs=len(fitted)))
    return dict(seed=seed.tolist(), tree_points=len(tree.data), trials=details,
                shared_pairs_at_seed=len(pairs[0] & pairs[1]),
                shared_accepted_pairs=len(final_pairs[0] & final_pairs[1]),
                local_xy_m=float(np.linalg.norm(np.subtract(details[0]['pose'], details[1]['pose'])[:2])))


def replay(stream, targets):
    states = {}
    original = slam.match

    def trace(xy, tree, seed):
        if index in targets:
            states[index] = (tree, seed.copy())
        return original(xy, tree, seed)

    poses, statuses = [], []
    with patch.object(slam, 'match', trace):
        tracker = slam.track(stream, [0., 0., 0.], grid=TrajectoryOnly())
        for index in range(len(stream)):
            row = next(tracker)
            poses.append(row.pose)
            statuses.append(row.status)
    return np.array(poses), np.array(statuses), states


def experiment(data, comparison):
    cfg = load_config(data/'profile.json')
    paths = [data/'mapping', comparison/'mapping']
    streams = [list(scans(path, cfg)) for path in paths]
    saved_rows = [saved(comparison/label/'map/trajectory.csv') for label in ('direct', 'converted')]
    poses = [np.array([[float(r[k]) for k in ('x', 'y', 'yaw')] for r in rows]) for rows in saved_rows]
    statuses = [np.array([r['status'] for r in rows]) for rows in saved_rows]
    delta = np.linalg.norm((poses[0]-poses[1])[:, :2], axis=1)
    first = {name: int(np.flatnonzero(mask)[0]) for name, mask in
             [('nonzero', delta > 0), ('1cm', delta > .01), ('10cm', delta > .1),
              ('status', statuses[0] != statuses[1])]}
    targets = sorted(set(first.values()))
    histories = []
    for stream, expected, status, rows in zip(streams, poses, statuses, saved_rows):
        np.testing.assert_array_equal([s.t for s in stream], [float(r['t']) for r in rows])
        actual, labels, states = replay(stream, targets)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(labels, status)
        histories.append(states)
    raws = [raw_scans(path, [0, *targets]) for path in paths]
    report = dict(first=first, frames={}, substitutions={})
    origin_ids = [representatives(r[0], s[0].xy) for r, s in zip(raws, streams)]
    report['origin'] = dict(raw=len(raws[0][0]), kept=[len(v) for v in origin_ids],
                            common=len(set(origin_ids[0]) & set(origin_ids[1])))
    for i in targets:
        xy = [s[i].xy for s in streams]
        raw = [r[i] for r in raws]
        ids = [representatives(r, s) for r, s in zip(raw, xy)]
        record = dict(t=streams[0][i].t, xy_m=delta[i], status=[str(s[i]) for s in statuses],
                      input_points=len(raw[0]), max_raw_xy_m=np.linalg.norm(raw[0]-raw[1], axis=1).max(),
                      kept=[len(v) for v in ids], shared_beams=len(set(ids[0]) & set(ids[1])))
        for name, history in zip(('direct_history', 'converted_history'), histories):
            tree, seed = history[i]
            record[name] = compare_matching(xy, ids, tree, seed)
        tree, seed = histories[0][i]
        base_pose = np.array(record['direct_history']['trials'][0]['pose'])
        for label, points in [('only_coordinates', raw[1][ids[0]]),
                              ('only_representatives', raw[0][ids[1]])]:
            pose, support, ok = slam.match(points, tree, seed)
            record[label] = dict(delta_xy_m=np.linalg.norm((pose-base_pose)[:2]),
                                 accepted=ok, support=support)
        report['frames'][i] = record
    for i in targets:
        stream = streams[0].copy()
        stream[i] = streams[1][i]
        mixed, status, _ = replay(stream, ())
        distance = np.linalg.norm((mixed-poses[0])[:, :2], axis=1)
        report['substitutions'][i] = dict(at_m=distance[i], next_m=distance[i+1],
                                          max_m=distance.max(), end_m=distance[-1],
                                          status_changes=int((status != statuses[0]).sum()))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'comparison', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = experiment(args.data, args.comparison)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
