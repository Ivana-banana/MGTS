"""Read a static TF tree and export the sensor extrinsics required by a profile."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .bag import reader


def collect(edges, item):
    parent, child = item.header.frame_id, item.child_frame_id
    if not parent or not child or child == parent:
        raise ValueError(f'Invalid static TF edge {parent!r} <- {child!r}')
    trans, q = item.transform.translation, item.transform.rotation
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    mat[:3, 3] = [trans.x, trans.y, trans.z]
    if not np.isfinite(mat).all():
        raise ValueError(f'Invalid transform for {child}')
    if child in edges:
        p, old = edges[child]
        if p != parent or not np.allclose(old, mat, atol=1e-6, rtol=0):
            raise ValueError(f'Conflicting static transforms for {child}')
    edges[child] = (parent, mat)


def root_from(frame, edges):
    visited = set()
    transform = np.eye(4)
    while frame in edges:
        if frame in visited:
            raise ValueError(f'Cycle in TF tree at {frame}')
        visited.add(frame)
        frame, parent_from_child = edges[frame]
        transform = parent_from_child @ transform
    return frame, transform


def extrinsics(edges, base, lidar, imu):
    root, root_from_base = root_from(base, edges)
    result = dict(base_frame=base, lidar_frame=lidar, imu_frame=imu)
    for sensor, frame in [('lidar', lidar), ('imu', imu)]:
        other_root, root_from_sensor = root_from(frame, edges)
        if root != other_root:
            raise ValueError(f'No static TF path from {frame!r} to {base!r}')
        base_from_sensor = np.linalg.solve(root_from_base, root_from_sensor)
        if sensor == 'lidar':
            result['base_from_lidar_xyz_m'] = base_from_sensor[:3, 3].tolist()
        result[f'base_from_{sensor}_xyzw'] = Rotation.from_matrix(base_from_sensor[:3, :3]).as_quat().tolist()
    return result


def read_edges(path, topic):
    edges = {}
    with reader(path) as bag:
        matches = [c for c in bag.connections if c.topic == topic and c.msgtype == 'tf2_msgs/msg/TFMessage']
        if not matches:
            raise ValueError(f'No TFMessage on {topic}; check the provided calibration files')
        for c, _, raw in bag.messages(connections=matches):
            for item in bag.deserialize(raw, c.msgtype).transforms:
                collect(edges, item)
    return edges


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python -m mgts_nav.calibration', description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--base', required=True)
    parser.add_argument('--lidar', required=True)
    parser.add_argument('--imu', required=True)
    parser.add_argument('--topic', default='/tf_static')
    args = parser.parse_args()
    try:
        print(json.dumps(extrinsics(read_edges(args.bag, args.topic), args.base, args.lidar, args.imu), indent=2))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'{exc}\n')
