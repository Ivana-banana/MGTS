"""Read a static TF tree and export the sensor extrinsics required by a profile."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .bag import cloud_xyz, connections, load_config, reader


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


def extrinsics(edges, base, lidar, imu=None):
    root, root_from_base = root_from(base, edges)
    result = dict(base_frame=base, lidar_frame=lidar, imu_frame=imu)
    for sensor, frame in [('lidar', lidar), ('imu', imu)]:
        if frame is None:
            continue
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


def fit_floor(xyz, start_quantile=20, tolerance=0.05, rounds=4):
    """Least-squares plane z = a*x + b*y + c through the lowest returns, refined on inliers."""
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) < 50:
        raise ValueError('Too few finite lidar points to look for the floor')
    keep = xyz[:, 2] <= np.percentile(xyz[:, 2], start_quantile)
    for _ in range(rounds):
        if keep.sum() < 30:
            break
        design = np.column_stack([xyz[keep, :2], np.ones(keep.sum())])
        a, b, c = np.linalg.lstsq(design, xyz[keep, 2], rcond=None)[0]
        residual = xyz[:, 2] - (a * xyz[:, 0] + b * xyz[:, 1] + c)
        keep = np.abs(residual) < tolerance
    if keep.sum() < 30:
        raise ValueError('Floor not observed')
    # Pitch is positive when the floor rises towards +x in the body frame (sensor pitched down
    # in the profile less than in reality), roll positive when it rises towards +y.
    return dict(floor_points=int(keep.sum()), floor_fraction=float(keep.mean()), floor_height_m=float(c),
                floor_rise_pitch_deg=float(np.degrees(np.arctan(a))),
                floor_rise_roll_deg=float(np.degrees(np.arctan(b))),
                floor_tilt_deg=float(np.degrees(np.arctan(np.hypot(a, b)))))


def floor_check(path, profile, scans=10):
    """Apply the profile lidar calibration to the first scans and fit the floor plane.

    With a correct calibration the floor is horizontal at z = 0 in the body frame (if the body
    frame is on the floor). A tilt of a few degrees means a missing or wrong lidar rotation.
    """
    cfg = load_config(profile)
    rot = Rotation.from_quat(cfg['base_from_lidar_xyzw'])
    offset = np.asarray(cfg['base_from_lidar_xyz_m'], dtype=float)
    clouds = []
    with reader(path) as bag:
        for conn, _, raw in bag.messages(connections=connections(bag, cfg)['lidar']):
            clouds.append(cloud_xyz(bag.deserialize(raw, conn.msgtype)))
            if len(clouds) == scans:
                break
    if not clouds:
        raise ValueError('No lidar scans in recording')
    xyz = rot.apply(np.concatenate(clouds).astype(float)) + offset
    try:
        result = fit_floor(xyz)
    except ValueError as exc:
        return dict(scans=len(clouds), floor_observed=False, reason=str(exc))
    if offset[2] - result['floor_height_m'] < 0.05 + 0.25 * abs(offset[2]):
        # A planar or nearly planar scan: the lowest returns are walls at sensor height.
        return dict(scans=len(clouds), floor_observed=False,
                    reason='Lowest returns lie near the sensor height; the floor is not visible')
    return dict(scans=len(clouds), floor_observed=True, **result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--base')
    parser.add_argument('--lidar')
    parser.add_argument('--imu', help='Omit when the robot has no IMU')
    parser.add_argument('--topic', default='/tf_static')
    parser.add_argument('--floor', type=Path, metavar='PROFILE',
                        help='Check the profile lidar calibration against the floor plane instead')
    args = parser.parse_args()
    try:
        if args.floor:
            print(json.dumps(floor_check(args.bag, args.floor), indent=2))
        elif not args.base or not args.lidar:
            parser.error('give --base and --lidar frames, or --floor PROFILE')
        else:
            print(json.dumps(extrinsics(read_edges(args.bag, args.topic), args.base, args.lidar, args.imu), indent=2))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'{exc}\n')
