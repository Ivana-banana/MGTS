"""Export an MGTS map and trajectory to a ROS 2 SQLite bag for RViz.

python -m mgts_nav.ros_export MAP.yaml TRAJECTORY.csv OUTPUT_BAG
ros2 bag play OUTPUT_BAG

Start RViz before playback. Set Fixed Frame to map and add Map (/map),
Pose (/pose), and Path (/path). /map arrives at the first pose, /pose advances
on the original CSV clock, and /path shows the full result at the last pose.
Select Transient Local durability for Map and Path for late subscription while
the player is alive. All headers use map coordinates, so no TF tree is needed.
"""
import argparse
import csv
import json
import math
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
from rosbags.interfaces import Qos, QosDurability, QosHistory, QosLiveliness, QosReliability, QosTime
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from .grid import read_map


def _read_poses(csv_path, types, frame):
    rows = []
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'t', 'x', 'y', 'yaw'}.issubset(reader.fieldnames or []):
            raise ValueError('Trajectory needs t,x,y,yaw columns')
        for line, row in enumerate(reader, 2):
            try:
                # Binary floats discard individual nanoseconds in epoch timestamps.
                t = Decimal(row['t'])
                if not t.is_finite() or not 0 <= t < 2**31:
                    raise ValueError('Timestamp outside supported ROS time range')
                ns = int((t * 1_000_000_000).to_integral_value())
                sec, nsec = divmod(ns, 1_000_000_000)
                if sec >= 2**31 or (rows and ns <= rows[-1][0]):
                    raise ValueError('Invalid or non-increasing timestamp after nanosecond rounding')
                x, y, yaw = (float(row[k]) for k in ('x', 'y', 'yaw'))
                if not all(map(math.isfinite, (x, y, yaw))):
                    raise ValueError('Non-finite pose')
            except (TypeError, ValueError, InvalidOperation) as exc:
                raise ValueError(f'{csv_path}:{line}: {exc}') from exc
            stamp = types['builtin_interfaces/msg/Time'](sec, nsec)
            header = types['std_msgs/msg/Header'](stamp, frame)
            pose = _pose(types, x, y, yaw)
            rows.append((ns, types['geometry_msgs/msg/PoseStamped'](header, pose)))
    if not rows:
        raise ValueError('Trajectory is empty')
    return rows


def _pose(types, x, y, yaw):
    position = types['geometry_msgs/msg/Point'](x, y, 0.)
    orientation = types['geometry_msgs/msg/Quaternion'](0., 0., math.sin(yaw / 2), math.cos(yaw / 2))
    return types['geometry_msgs/msg/Pose'](position, orientation)


def _profile(durability, depth):
    default = QosTime(0, 0)
    return Qos(QosHistory.KEEP_LAST, depth, QosReliability.RELIABLE, durability, default,
               default, QosLiveliness.AUTOMATIC, default, False)


def export(map_yaml, trajectory, output, frame='map'):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    if not frame or frame.startswith('/') or any(c.isspace() for c in frame):
        raise ValueError('Expected a non-empty frame name without whitespace or leading slash')
    pixels, meta, _ = read_map(map_yaml)
    store = get_typestore(Stores.ROS2_HUMBLE)
    types = store.types
    rows = _read_poses(trajectory, types, frame)
    first_ns, first_pose = rows[0]
    last_ns, last_pose = rows[-1]
    # A PGM starts at the top left. OccupancyGrid row zero lies at map origin.
    probability = 1. - np.arange(256) / 255.
    classes = np.full(256, -1, dtype=np.int8)
    classes[probability < meta['free_thresh']] = 0
    classes[probability > meta['occupied_thresh']] = 100
    occupancy = classes[np.flipud(pixels)]
    height, width = pixels.shape
    metadata = types['nav_msgs/msg/MapMetaData'](first_pose.header.stamp, meta['resolution'],
                                                  width, height, _pose(types, *meta['origin']))
    grid = types['nav_msgs/msg/OccupancyGrid'](first_pose.header, metadata, occupancy.ravel())
    path = types['nav_msgs/msg/Path'](last_pose.header, [pose for _, pose in rows])
    latched = _profile(QosDurability.TRANSIENT_LOCAL, 1)
    live = _profile(QosDurability.VOLATILE, 10)
    output.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp) / output.name
        with Writer(staged, version=8) as writer:
            grid_conn = writer.add_connection('/map', grid.__msgtype__, typestore=store, offered_qos_profiles=[latched])
            pose_conn = writer.add_connection('/pose', first_pose.__msgtype__, typestore=store, offered_qos_profiles=[live])
            path_conn = writer.add_connection('/path', path.__msgtype__, typestore=store, offered_qos_profiles=[latched])
            writer.write(grid_conn, first_ns, store.serialize_cdr(grid, grid_conn.msgtype))
            for ns, pose in rows:
                writer.write(pose_conn, ns, store.serialize_cdr(pose, pose_conn.msgtype))
            writer.write(path_conn, last_ns, store.serialize_cdr(path, path_conn.msgtype))
        staged.rename(output)
    return dict(bag=str(output), poses=len(rows), frame=frame, start_ns=first_ns, end_ns=last_ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('map_yaml', type=Path)
    parser.add_argument('trajectory_csv', type=Path)
    parser.add_argument('output_bag', type=Path)
    parser.add_argument('--frame', default='map', help='Frame shared by map and trajectory')
    args = parser.parse_args()
    try:
        result = export(args.map_yaml, args.trajectory_csv, args.output_bag, args.frame)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
