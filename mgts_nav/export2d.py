"""Export the final Go2 3D result as a 2D occupancy grid and a standard trajectory.

python -m mgts_nav.export2d MAPPING TRAJECTORY --out DIR

MAPPING is the walk-through recording that built map.npz in python -m mgts_nav.localize3d.
The grid uses the same deskewed scans and odometry poses, so it lies in the frame of map.npz
and of the final poses. TRAJECTORY is trajectory.csv from python -m mgts_nav.localize3d or the
final rosbag2 directory with geometry_msgs/PoseStamped on /mgts/pose3d.
Writes map.pgm + map.yaml (ROS map_server), trajectory.tum (evo), trajectory.csv
(t,x,y,yaw,status for python -m mgts_nav.metrics), route.png and deliverables.json.
"""
import argparse
import csv
import hashlib
import json
import math
import shlex
import sys
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .grid import Grid, check_resolution, preview
from .localize3d import alignment_argument, alignment_pose, frame_arguments, scans

# Same as height_band_m in configs/go2-official.json; the image ships only mgts_nav/*.py.
HEIGHT_BAND_M = (-0.15, 1.5)
RESOLUTION_M = 0.05
POSE_TOPIC = '/mgts/pose3d'


class Row(NamedTuple):
    stamp: str  # exact decimal text of the source timestamp
    t: float
    xyz: tuple
    quat: tuple  # x, y, z, w
    status: str

    @property
    def pose(self):
        return self.xyz[0], self.xyz[1], yaw(self.quat)


def yaw(quat):
    """Heading about the map Z axis from an x, y, z, w quaternion (intrinsic ZYX)."""
    return float(Rotation.from_quat(quat).as_euler('ZYX')[0])


def build_grid(clouds, resolution=RESOLUTION_M, band=HEIGHT_BAND_M, every=1):
    """Accumulate (points_base, pose) pairs; pose.apply maps base into the map frame."""
    low, high = band
    if not low < high:
        raise ValueError('height band must have LOW < HIGH metres')
    if every < 1:
        raise ValueError('--every must be a positive integer')
    grid = Grid(check_resolution(resolution))
    used = total = 0
    for i, (points, pose) in enumerate(clouds):
        total += 1
        if i % every:
            continue
        points = points[(points[:, 2] >= low) & (points[:, 2] <= high)]
        if len(points):
            grid.observe(np.asarray(pose.xyz[:2], dtype=float), pose.apply(points)[:, :2])
            used += 1
    return grid, used, total


def check_rows(rows, source):
    if not rows:
        raise ValueError(f'{source}: no poses')
    for previous, row in zip(rows, rows[1:]):
        if not row.t > previous.t:
            raise ValueError(f'{source}: timestamps must increase ({previous.stamp} then {row.stamp})')
    for row in rows:
        values = (row.t, *row.xyz, *row.quat)
        if not all(map(math.isfinite, values)) or not np.linalg.norm(row.quat) > 0:
            raise ValueError(f'{source}: invalid pose at t={row.stamp}')
    return rows


def read_csv(path):
    rows = []
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = ['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        missing = [key for key in required if key not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f'{path}: missing columns {",".join(missing)}; expected '
                             'trajectory.csv from python -m mgts_nav.localize3d')
        for line, row in enumerate(reader, 2):
            try:
                values = [float(row[key]) for key in required]
            except (TypeError, ValueError) as error:
                raise ValueError(f'{path}:{line}: invalid number') from error
            rows.append(Row(row['t'].strip(), values[0], tuple(values[1:4]), tuple(values[4:]),
                            row.get('status') or 'unknown'))
    return check_rows(rows, path), 'map'


def read_bag(path, topic=POSE_TOPIC):
    from .bag import reader

    rows, frames = [], set()
    with reader(path) as bag:
        connections = [c for c in bag.connections if c.topic == topic]
        if not connections:
            found = ', '.join(sorted({c.topic for c in bag.connections})) or 'none'
            raise ValueError(f'{path}: no {topic} topic; found {found}')
        for connection, _, raw in bag.messages(connections=connections):
            if connection.msgtype != 'geometry_msgs/msg/PoseStamped':
                raise ValueError(f'{path}: {topic} has type {connection.msgtype}, '
                                 'expected geometry_msgs/msg/PoseStamped')
            message = bag.deserialize(raw, connection.msgtype)
            s, p, q = message.header.stamp, message.pose.position, message.pose.orientation
            frames.add(message.header.frame_id)
            rows.append(Row(f'{s.sec}.{s.nanosec:09d}', s.sec + s.nanosec * 1e-9,
                            (p.x, p.y, p.z), (q.x, q.y, q.z, q.w), 'published'))
    if len(frames) > 1:
        raise ValueError(f'{path}: {topic} mixes frames {sorted(frames)}')
    return check_rows(rows, path), next(iter(frames), '')


def read_trajectory(path):
    path = Path(path)
    return read_bag(path) if path.is_dir() else read_csv(path)


def write_trajectory(rows, directory):
    with (directory / 'trajectory.tum').open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(' '.join([row.stamp, *map(repr, map(float, (*row.xyz, *row.quat)))]) + '\n')
    with (directory / 'trajectory.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['t', 'x', 'y', 'yaw', 'status'])
        for row in rows:
            writer.writerow([row.stamp, *map(repr, map(float, row.pose)), row.status])


def digest(path):
    path = Path(path)
    files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    return {str(p.relative_to(path) if path.is_dir() else p.name):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mapping', type=Path, help='Walk-through rosbag2 that built map.npz')
    parser.add_argument('trajectory', type=Path,
                        help='localize3d trajectory.csv or final rosbag2 directory with /mgts/pose3d')
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    parser.add_argument('--resolution', type=float, default=RESOLUTION_M, help='Cell size, metres')
    parser.add_argument('--height-band', type=float, nargs=2, default=HEIGHT_BAND_M,
                        metavar=('LOW', 'HIGH'), help='Base-frame point heights kept, metres')
    parser.add_argument('--every', type=int, default=1, help='Use every K-th mapping scan')
    alignment_argument(parser)
    parser.epilog = ('--map-from-odom maps the odom frame of MAPPING into the trajectory map frame, '
                     'as localize3d does for inspection poses. The final run built map.npz from '
                     'the mapping odom frame, so omit it there.')
    frame_arguments(parser)
    args = parser.parse_args()
    started = perf_counter()
    try:
        alignment = alignment_pose(args.map_from_odom)
        rows, frame = read_trajectory(args.trajectory)
        args.out.mkdir(parents=True)
        frames = (args.cloud_frame, args.odom_frame, args.odom_child_frame)
        clouds = ((points, pose if alignment is None else alignment.compose(pose))
                  for _, points, pose, _ in scans(args.mapping, *frames))
        grid, used, total = build_grid(clouds, args.resolution, args.height_band, args.every)
        grid.save(args.out)
    except (OSError, ValueError) as error:
        parser.exit(2, f'{error}\n')
    write_trajectory(rows, args.out)
    preview(args.out / 'map.yaml', args.out / 'route.png', rows)
    pixels = np.array(Image.open(args.out / 'map.pgm'))
    summary = {
        'frame': frame or 'map',
        'map_frame_note': 'grid and poses share the frame of map.npz (mapping odometry)'
                          + ('' if alignment is None else ' after --map-from-odom'),
        'map_from_odom': args.map_from_odom,
        'resolution_m': args.resolution,
        'height_band_base_m': list(args.height_band),
        'scan_stride': args.every,
        'map_scans_total': total,
        'map_scans_used': used,
        'poses': len(rows),
        'pose_status': {s: sum(r.status == s for r in rows) for s in sorted({r.status for r in rows})},
        'grid_size_cells': [int(pixels.shape[1]), int(pixels.shape[0])],
        'cells': {'occupied': int((pixels == 0).sum()), 'free': int((pixels == 254).sum()),
                  'unknown': int((pixels == 205).sum())},
        'trajectory_sha256': digest(args.trajectory),
        'elapsed_s': round(perf_counter() - started, 2),
        'command': shlex.join(['python', '-m', 'mgts_nav.export2d', *sys.argv[1:]]),
    }
    (args.out / 'deliverables.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
