"""Audit corrected ROS header times against the interpolation rules used by navigation."""
import argparse
from array import array
import json
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReaderError
from rosbags.serde.errors import SerdeError
from rosbags.typesys.base import TypesysError

from .bag import connections, interval_samples, load_config, reader, stamp


def _read_clocks(path, cfg):
    buffers = {role: array('d') for role in ('lidar', 'imu', 'odom')}
    headers = {}
    with reader(path) as bag:
        selected = connections(bag, cfg)
        roles = {c.topic: role for role, conns in selected.items() for c in conns}
        conns = [c for group in selected.values() for c in group]
        for conn, _, raw in bag.messages(connections=conns):
            role = roles[conn.topic]
            try:
                header = stamp(bag.deserialize(raw, conn.msgtype))
            except (TypesysError, SerdeError, ValueError, KeyError) as exc:
                raise ValueError(f'Cannot decode {conn.topic}: {exc}') from exc
            t = header + cfg.get('header_offset_s', {}).get(role, 0.)
            if not np.isfinite(t):
                raise ValueError(f'{conn.topic}: non-finite corrected header time')
            values = buffers[role]
            if values and t <= values[-1]:
                reason = 'duplicate' if t == values[-1] else 'backwards'
                raise ValueError(f'{conn.topic}: {reason} timestamp at message {len(values) + 1}: {values[-1]} -> {t}')
            values.append(t)
            if role not in headers:
                headers[role] = [header, header]
            else:
                headers[role][1] = header
    return {role: np.asarray(values) for role, values in buffers.items()}, headers


def _coverage(clock, starts, ends, gap):
    missing = {}
    covered = 0
    for start, end in zip(starts, ends, strict=True):
        if interval_samples(clock, start, end, gap) is not None:
            covered += 1
            continue
        if not len(clock):
            reason = 'no_messages'
        elif start < clock[0] and end > clock[-1]:
            reason = 'outside_both_edges'
        elif start < clock[0]:
            reason = 'before_start'
        elif end > clock[-1]:
            reason = 'after_end'
        else:
            reason = 'internal_gap'
        span = missing.setdefault(reason, dict(count=0, first=[float(start), float(end)]))
        span['count'] += 1
        span['last'] = [float(start), float(end)]
    total = len(starts)
    if not total:
        status = 'no_intervals'
    elif covered == total:
        status = 'complete'
    else:
        status = 'partial' if covered else 'none'
    return dict(status=status, covered=covered, total=total, missing=missing)


def audit(path, cfg):
    clocks, headers = _read_clocks(path, cfg)
    lidar = clocks['lidar']
    if not len(lidar):
        raise ValueError(f"No lidar messages on {cfg['lidar']}")
    gap = cfg['max_sync_gap_s']
    report = dict(scope='header_time_interpolation', max_sync_gap_s=gap, clocks={}, coverage={})
    for role, times in clocks.items():
        report['clocks'][role] = dict(topic=cfg[role], messages=len(times), header_range_s=headers.get(role),
                                      offset_s=cfg.get('header_offset_s', {}).get(role, 0.),
                                      corrected_range_s=[float(times[0]), float(times[-1])] if len(times) else None)
    for role, starts, ends in [('odom', lidar, lidar), ('imu', lidar[:-1], lidar[1:])]:
        if cfg[role] is None:
            coverage = dict(status='disabled')
        else:
            coverage = _coverage(clocks[role], starts, ends, gap)
        report['coverage'][role] = coverage
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--require-complete', action='store_true',
                        help='Exit 3 unless every enabled motion channel covers all required lidar times')
    args = parser.parse_args()
    try:
        report = audit(args.bag, load_config(args.config))
    except (AnyReaderError, OSError, ValueError) as exc:
        parser.exit(2, f'{exc}\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.require_complete and any(item['status'] not in ('complete', 'disabled')
                                     for item in report['coverage'].values()):
        parser.exit(3)


if __name__ == '__main__':
    main()
