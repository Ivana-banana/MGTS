"""Estimate field yaw from RGB-D and assumed speed ceilings, then score against mocap."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from mgts_nav.image_pairs import tracks
from mgts_nav.vision import estimate
from tools.tum_visual_yaw import K, MOUNT, depth_tracks, frame, rectification


SPEEDS = (.02, .10, .50)


def images(table):
    rows = [line.split() for line in table.read_text(encoding='ascii').splitlines()
            if line.strip() and not line.startswith('#')]
    return np.array([float(t) for t, _ in rows]), [table.parent/name for _, name in rows]


def predict(rgb_table, depth_table):
    rt, rgb = images(rgb_table)
    dt, depths = images(depth_table)
    coords, valid = rectification()
    quat = Rotation.from_matrix(MOUNT).as_quat().tolist()
    rows = []
    for i in range(0, len(rgb)-6, 10):
        first = frame(rgb[i], coords, valid)
        j = int(np.argmin(abs(dt-rt[i])))
        with Image.open(depths[j]) as image:
            depth = np.array(image)
        for gap, max_dt in [(1, .10), (6, .40)]:
            elapsed = rt[i+gap]-rt[i]
            row = dict(index=i, gap=gap, t0=rt[i], t1=rt[i+gap], trials=[])
            rows.append(row)
            if not 0 < elapsed <= max_dt or abs(dt[j]-rt[i]) > .02:
                reason = 'rgb_interval' if not 0 < elapsed <= max_dt else 'depth_gap'
                row['trials'] = [dict(speed=v, accepted=False, reason=reason) for v in SPEEDS]
                continue
            matched = tracks(first, frame(rgb[i+gap], coords, valid))
            kept, distance = depth_tracks(matched, depth, coords)
            row['range_bound_m'] = distance
            for speed in SPEEDS:
                bound = speed*elapsed
                result = estimate(dict(K=K.tolist(), base_from_camera_xyzw=quat, tracks=kept,
                                       camera_motion_bound_m=bound, min_scene_range_m=distance))
                row['trials'].append(dict(speed=speed, motion_bound_m=bound, **result))
    return rows


def score(rows, reference):
    ref = np.loadtxt(reference)
    times = ref[:, 0]
    rotations = Slerp(times, Rotation.from_quat(ref[:, 4:8]))
    for row in rows:
        t = np.array([row['t0'], row['t1']])
        hi = np.searchsorted(times, t)
        if hi.min() == 0 or hi.max() >= len(times) or np.any(times[hi]-times[hi-1] > .04):
            continue
        position = np.array([np.interp(t, times, ref[:, col]) for col in (1, 2, 3)])
        displacement = float(np.linalg.norm(position[:, 1]-position[:, 0]))
        r0, r1 = rotations(t).as_matrix()
        delta = MOUNT @ r0.T @ r1 @ MOUNT.T
        yaw = float(np.arctan2(delta[1, 0], delta[0, 0]))
        tilt = float(np.degrees(np.arccos(np.clip(delta[2, 2], -1, 1))))
        row.update(net_displacement_m=displacement, tilt_deg=tilt, reference_yaw_rad=yaw)
        for trial in row['trials']:
            trial['violated'] = bool(displacement > trial['speed']*(t[1]-t[0]))
            if trial['accepted']:
                error = trial['yaw_rad']-yaw
                trial['error_deg'] = float(np.degrees(np.arctan2(np.sin(error), np.cos(error))))
    summary = []
    for gap in (1, 6):
        group = [r for r in rows if r['gap'] == gap]
        for index, speed in enumerate(SPEEDS):
            trials = [r['trials'][index] for r in group]
            scored = [q for q in trials if 'violated' in q]
            accepted = [q for q in scored if q['accepted']]
            errors = np.array([q['error_deg'] for q in accepted])
            summary.append(dict(gap=gap, speed=speed, total=len(trials), reasons=Counter(q['reason'] for q in trials),
                                accepted=sum(q['accepted'] for q in trials), scored=len(scored),
                                accepted_scored=len(accepted), violations=sum(q['violated'] for q in scored),
                                accepted_violations=sum(q['violated'] for q in accepted),
                                accepted_nonplanar=sum(r['trials'][index]['accepted'] and r.get('tilt_deg', 0) > 3
                                                       for r in group),
                                rms_deg=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                                max_abs_deg=float(np.max(abs(errors))) if len(errors) else None))
    return dict(summary=summary, pairs=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    blind = commands.add_parser('predict')
    blind.add_argument('rgb', type=Path)
    blind.add_argument('depth', type=Path)
    scored = commands.add_parser('score')
    scored.add_argument('predictions', type=Path)
    scored.add_argument('reference', type=Path)
    for command in (blind, scored):
        command.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'predict':
        result = predict(args.rgb, args.depth)
    else:
        result = score(json.loads(args.predictions.read_bytes()), args.reference)
        print(json.dumps(result['summary'], indent=2))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
