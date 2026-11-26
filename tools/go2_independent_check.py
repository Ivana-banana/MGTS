"""Compare Go2 trajectories on preregistered windows of the official inspection."""
import argparse
import csv
import json
from itertools import groupby
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from mgts_nav.bag import load_config, reader, scans as scans2d, stamp
from mgts_nav.grid import read_map
from mgts_nav.localize3d import Motion, Pose, odometry_sample, scans
from mgts_nav.slam import relative, track, world


METHODS = ('odom', '2d', '3d')
WINDOWS = {'all': (0, 110), 'start': (0, 10), 'turn': (38, 46), 'end': (99, 110),
           'historical_gap': (40.5, 110)}


def planar(pose):
    return np.r_[pose.xyz[:2], pose.rotation.as_euler('xyz')[2]]


def map_origin(mapping):
    with reader(mapping) as bag:
        lidar = [c for c in bag.connections if c.topic == '/utlidar/cloud_base']
        c, _, raw = next(bag.messages(connections=lidar))
        start = stamp(bag.deserialize(raw, c.msgtype))
        samples, times = [], []
        for c, _, raw in bag.messages(connections=[c for c in bag.connections if c.topic == '/utlidar/robot_odom']):
            msg = bag.deserialize(raw, c.msgtype)
            samples.append(odometry_sample(msg))
            times.append(stamp(msg))
            if len(times) >= 2 and times[-1] >= start:
                break
        use = np.clip(start, times[0], times[-1])
        return planar(Motion(times, *zip(*samples)).at(use)), float(use - start)


def live_poses(result):
    poses = {}
    with reader(result) as bag:
        for c, _, raw in bag.messages(connections=[c for c in bag.connections if c.topic == '/mgts/pose3d']):
            msg = bag.deserialize(raw, c.msgtype)
            p, q = msg.pose.position, msg.pose.orientation
            poses[stamp(msg)] = Pose(Rotation.from_quat([q.x, q.y, q.z, q.w]), np.array([p.x, p.y, p.z]))
    return poses


def between_clouds(a, b):
    ab = cKDTree(a).query(b)[0]
    ba = cKDTree(b).query(a)[0]
    return float((np.median(ab) + np.median(ba)) / 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mapping', 'inspection', 'config', 'map2d', 'map3d', 'live', 'timings'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--initial-2d', type=float, nargs=3, metavar=('X', 'Y', 'YAW'),
                        help='explicit 2D start in its map frame, checked against the recorded 3D prior')
    args = parser.parse_args()
    with args.timings.open() as stream:
        timings = {float(row['header_s']): row['status'] for row in csv.DictReader(stream)}
    input_times = sorted(timings)
    origin, origin_dt = map_origin(args.mapping)
    poses3d = live_poses(args.live)
    first, _, prior, _ = next(scans(args.inspection))
    start = relative(origin, planar(prior))
    if args.initial_2d is not None:
        if not np.allclose(args.initial_2d, start, atol=1e-8, rtol=0):
            parser.error('--initial-2d differs from the recorded common start')
        start = np.array(args.initial_2d)
    sliced = (s for s in scans2d(args.inspection, load_config(args.config)) if s.t >= first)
    rows2d = {r.t: r for r in track(sliced, start, map_points=read_map(args.map2d)[2])}
    with np.load(args.map3d) as data:
        tree = cKDTree(data['xyz'])
    records, samples = [], []
    previous_t, previous_xy = None, {}
    next_sample = 0
    for t, cloud, odom, _ in scans(args.inspection):
        elapsed = t - input_times[0]
        row = rows2d[t]
        angles = odom.rotation.as_euler('xyz')
        angles[2] = origin[2] + row.pose[2]
        xy = world(row.pose[:2], origin)
        p2 = Pose(Rotation.from_euler('xyz', angles), np.r_[xy, odom.xyz[2]])
        poses = {'odom': odom, '2d': p2, '3d': poses3d[t]}
        points = {k: p.apply(cloud[::4]) for k, p in poses.items()}
        record = dict(t=t, elapsed=elapsed)
        for k, pose in poses.items():
            error = float(np.median(tree.query(points[k])[0]))
            jump = previous_t is not None and t-previous_t <= .15 and np.linalg.norm(pose.xyz[:2]-previous_xy[k]) > .5
            matched = {'odom': False, '2d': row.status=='matched', '3d': timings[t]=='matched'}[k]
            record[k] = (error, bool(jump), matched)
            previous_xy[k] = pose.xyz[:2]
        if not records:
            first_deltas = {k: relative(planar(odom), planar(p)).tolist() for k,p in poses.items()}
        records.append(record)
        previous_t = t
        if elapsed >= next_sample:
            samples.append((elapsed, planar(odom), points))
            next_sample = int(elapsed) + 1
    stats = {}
    for name,(lo,hi) in WINDOWS.items():
        subset = [r for r in records if lo <= r['elapsed'] < hi]
        stats[name] = {k: dict(n=len(subset), median_m=float(np.median([r[k][0] for r in subset])),
                              high=sum(r[k][0]>.30 for r in subset), jumps=sum(r[k][1] for r in subset),
                              matched=sum(r[k][2] for r in subset),
                              flagged_matched=sum(r[k][2] and (r[k][0]>.30 or r[k][1]) for r in subset))
                       for k in METHODS}
    gaps = {}
    for k in ('2d','3d'):
        matched = {r['t'] for r in records if r[k][2]}
        runs = []
        for ok, group in groupby(range(len(input_times)), key=lambda i: input_times[i] in matched):
            indices = list(group)
            if not ok:
                begin, end = indices[0], indices[-1]
                runs.append((input_times[min(end+1,len(input_times)-1)]-input_times[begin],len(indices),
                             input_times[begin]-input_times[0]))
        gaps[k] = max(runs,default=(0,0,0))
    pairs = []
    for i,(tb,pb,b) in enumerate(samples):
        candidates = []
        for j,(ta,pa,_) in enumerate(samples[:i]):
            delta = relative(pa,pb)
            distance = np.linalg.norm(delta[:2])
            if tb-ta >= 15 and distance <= .75 and abs(delta[2]) <= np.deg2rad(30):
                candidates.append((distance,j))
        if candidates:
            _,j = min(candidates)
            ta,_,a = samples[j]
            pairs.append(dict(times=[ta,tb], **{k:between_clouds(a[k],b[k]) for k in METHODS}))
    print(json.dumps(dict(input_scans=len(input_times),origin=origin.tolist(),origin_offset_s=origin_dt,
                          initial_2d=start.tolist(),first_outputs_relative_to_odom=first_deltas,
                          stats=stats,gaps=gaps,revisits=pairs),indent=2))


if __name__ == '__main__':
    main()
