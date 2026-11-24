"""Reconstruct a higher-rate planar inspection trajectory from matched scans and kinematics.

python -m mgts_nav.continuous BAG SCAN_TRAJECTORY.csv OUTPUT.csv --config PROFILE.json

The input CSV must belong to this inspection bag, in map coordinates and on its
corrected header clock. Only matched LiDAR poses anchor the predictions. Every
prediction is evaluated at a real odometry timestamp. There is no extrapolation
of the initial anchor, backfilling from a future match, or crossing an odometry
gap wider than the profile's max_sync_gap_s. After such a gap, predictions
resume at the next match with usable kinematics. Matched poses are still emitted
when odometry is missing. CSV status and anchor_age_s distinguish predictions
from scan matches. Output never feeds mapping. Ground truth is not an input.
"""
import argparse
import csv
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

from .bag import check_frame, load_config, reader, stamp
from .slam import relative, world


class Sample(NamedTuple):
    t: float
    x: float
    y: float
    yaw: float
    status: str
    anchor_age_s: float


def read_anchors(path):
    anchors = []
    previous = -np.inf
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = csv.DictReader(stream)
        if not {'t', 'x', 'y', 'yaw', 'status'}.issubset(rows.fieldnames or []):
            raise ValueError('Scan trajectory needs t,x,y,yaw,status')
        for line, row in enumerate(rows, 2):
            try:
                values = [float(row[k]) for k in ('t', 'x', 'y', 'yaw')]
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{path}:{line}: invalid number or missing field in t,x,y,yaw') from exc
            if not np.isfinite(values).all() or values[0] <= previous:
                raise ValueError(f'{path}:{line}: invalid scan pose or non-increasing time')
            previous = values[0]
            if row['status'] == 'matched':
                anchors.append(values)
            elif row['status'] not in ('prediction_only', 'origin'):
                raise ValueError(f'{path}:{line}: unknown status {row["status"]!r}')
    if not anchors:
        raise ValueError('Inspection has no matched scan anchors')
    return np.asarray(anchors)


def read_odometry(path, cfg):
    rows = []
    clock = cfg.get('header_offset_s', {}).get('odom', 0.)
    with reader(path) as source:
        conns = [c for c in source.connections if c.topic == cfg['odom']]
        if not conns or any(c.msgtype != 'nav_msgs/msg/Odometry' for c in conns):
            raise ValueError(f'Expected nav_msgs/msg/Odometry on {cfg["odom"]}')
        for conn, _, raw in source.messages(connections=conns):
            msg = source.deserialize(raw, conn.msgtype)
            check_frame(msg.header.frame_id, cfg['odom_frame'])
            check_frame(msg.child_frame_id, cfg['base_frame'])
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            forward = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()[:, 0]
            rows.append([stamp(msg) + clock, p.x, p.y, np.arctan2(forward[1], forward[0])])
    return np.asarray(rows, dtype=float).reshape(-1, 4)


def reconstruct(anchors, odom, gap):
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError('Expected positive max_sync_gap_s')
    for data in (anchors, odom):
        if (data.ndim != 2 or data.shape[1] != 4 or not np.isfinite(data).all()
                or np.any(np.diff(data[:, 0]) <= 0)):
            raise ValueError('Expected finite ordered t,x,y,yaw records')
    if not len(anchors):
        raise ValueError('No matched scan anchors')
    times = odom[:, 0]
    gaps = np.flatnonzero(np.diff(times) > gap)
    segments = np.cumsum(np.r_[False, np.diff(times) > gap])
    a = i = 0
    anchor_t = anchor_pose = anchor_local = anchor_segment = None
    skipped = 0
    result = []
    while a < len(anchors) or i < len(odom):
        if a < len(anchors) and (i == len(odom) or anchors[a, 0] <= times[i]):
            anchor_t, anchor_pose = anchors[a, 0], anchors[a, 1:]
            result.append(Sample(anchor_t, *anchor_pose, 'matched', 0.))
            anchor_local = anchor_segment = None
            if i < len(odom) and times[i] == anchor_t:
                anchor_local, anchor_segment = odom[i, 1:], segments[i]
                i += 1
            elif 0 < i < len(odom) and segments[i-1] == segments[i]:
                left, right = odom[i-1], odom[i]
                weight = (anchor_t - left[0]) / (right[0] - left[0])
                direction = right[3] - left[3]
                angle = left[3] + weight * np.arctan2(np.sin(direction), np.cos(direction))
                anchor_local = np.r_[left[1:3] + weight * (right[1:3] - left[1:3]), angle]
                anchor_segment = segments[i]
            a += 1
        else:
            if anchor_local is not None and anchor_segment == segments[i]:
                motion = relative(anchor_local, odom[i, 1:])
                xy = world(motion[:2], anchor_pose)
                result.append(Sample(times[i], *xy, anchor_pose[2] + motion[2],
                                     'prediction_only', times[i] - anchor_t))
            else:
                skipped += 1
            i += 1
    return result, dict(matched_anchors=len(anchors), odom_samples=len(times),
                        prediction_samples=len(result) - len(anchors), skipped_odom_samples=skipped,
                        max_prediction_age_s=float(max(row.anchor_age_s for row in result)),
                        max_anchor_interval_s=float(np.diff(anchors[:, 0]).max()) if len(anchors) > 1 else None,
                        odom_gaps_s=[[float(times[j]), float(times[j+1])] for j in gaps])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag', type=Path)
    parser.add_argument('scan_csv', type=Path)
    parser.add_argument('output_csv', type=Path)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        rows, summary = reconstruct(read_anchors(args.scan_csv), read_odometry(args.bag, cfg),
                                    cfg['max_sync_gap_s'])
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(Sample._fields)
            writer.writerows(rows)
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(2, f'{exc}\n')
    print(json.dumps(summary, allow_nan=False))


if __name__ == '__main__':
    main()
