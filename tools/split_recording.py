"""Find stops in one combined recording and cut it into mapping and inspection parts.

    python tools/split_recording.py BAG --stops /odom
    python tools/split_recording.py BAG --from 500 --to 520.5 --out data/mapping

`--stops` prints intervals where the odometry pose stays still (header time, seconds).
A cut keeps every message whose header stamp lies in [from, to]; messages without a
header and every /tf_static message are copied as is, so static calibration survives.
The output is a new ROS 2 bag (SQLite) that all mgts_nav commands accept.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mgts_nav.bag import reader, stamp  # noqa: E402


def stops(path, topic, min_s=1.0, speed=0.02, turn=0.02):
    """Still intervals of an Odometry topic: [start, end] header seconds lasting >= min_s."""
    rows = []
    with reader(path) as bag:
        conns = [c for c in bag.connections if c.topic == topic]
        if not conns:
            raise ValueError(f'No topic {topic}')
        for conn, _, raw in bag.messages(connections=conns):
            m = bag.deserialize(raw, conn.msgtype)
            p, q = m.pose.pose.position, m.pose.pose.orientation
            rows.append([stamp(m), p.x, p.y, 2 * np.arctan2(q.z, q.w)])
    a = np.asarray(rows)
    if len(a) < 2:
        return []
    dt = np.diff(a[:, 0])
    dt[dt <= 0] = np.nan
    v = np.hypot(*np.diff(a[:, 1:3], axis=0).T) / dt
    w = np.abs(np.angle(np.exp(1j * np.diff(a[:, 3])))) / dt
    still = (v < speed) & (w < turn)
    result, begin = [], None
    for i, flag in enumerate(still):
        if flag and begin is None:
            begin = a[i, 0]
        if (not flag or i == len(still) - 1) and begin is not None:
            end = a[i + 1, 0] if flag else a[i, 0]
            if end - begin >= min_s:
                result.append([float(begin), float(end)])
            begin = None
    return result


def cut(path, output, start, end, tf_static='/tf_static'):
    store = get_typestore(Stores.ROS2_HUMBLE)
    counts = {}
    with reader(path) as bag, Writer(output, version=8) as writer:
        conns = {}
        for conn, received, raw in bag.messages():
            keep = conn.topic == tf_static
            if not keep:
                try:
                    msg = bag.deserialize(raw, conn.msgtype)
                except Exception:  # unknown schema: keep by record time
                    keep = start <= received * 1e-9 <= end
                else:
                    keep = start <= stamp(msg) <= end if hasattr(msg, 'header') else True
            if not keep:
                continue
            if conn.id not in conns:
                conns[conn.id] = writer.add_connection(conn.topic, conn.msgtype, typestore=store)
                counts[conn.topic] = 0
            writer.write(conns[conn.id], received, raw)
            counts[conn.topic] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--stops', metavar='ODOM_TOPIC', help='List still intervals of this Odometry topic')
    parser.add_argument('--min-stop-s', type=float, default=1.0)
    parser.add_argument('--from', dest='start', type=float, help='Header time, seconds')
    parser.add_argument('--to', dest='end', type=float, help='Header time, seconds')
    parser.add_argument('--out', type=Path, help='New bag directory')
    args = parser.parse_args()
    try:
        if args.stops:
            print(json.dumps(stops(args.bag, args.stops, args.min_stop_s), indent=2))
            return
        if None in (args.start, args.end, args.out) or args.start >= args.end:
            parser.error('a cut needs --from < --to and --out')
        if args.out.exists():
            parser.error(f'{args.out} exists; choose a new directory')
        print(json.dumps(cut(args.bag, args.out, args.start, args.end), indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
