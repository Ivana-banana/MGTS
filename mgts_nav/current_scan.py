"""Add /current_scan to a ROS 2 result bag using exact, accepted tracker poses."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

from .bag import cloud_xyz, reader, stamp
from .result_bag_audit import timestamp_ns, topic_connections


def transformed_cloud(scan, pose, types):
    if scan.header.frame_id != 'base_link' or pose.header.frame_id != 'map':
        raise ValueError('Expected cloud_base in base_link and a tracker pose in map')
    if timestamp_ns(scan.header) != timestamp_ns(pose.header):
        raise ValueError('Scan and pose must have the same header stamp')
    p, q = pose.pose.position, pose.pose.orientation
    xyz = np.array([p.x, p.y, p.z])
    quat = [q.x, q.y, q.z, q.w]
    if not np.isfinite(xyz).all() or not np.isclose(np.linalg.norm(quat), 1, atol=1e-4, rtol=0):
        raise ValueError('Invalid tracker pose')
    points = cloud_xyz(scan)
    points = points[np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1)]
    points = (points @ Rotation.from_quat(quat).as_matrix().T + xyz).astype('<f4')
    return xyz_cloud(points, pose.header, types)


def xyz_cloud(points, header, types):
    points = np.ascontiguousarray(points, dtype='<f4')
    fields = [types['sensor_msgs/msg/PointField'](name, i * 4, 7, 1) for i, name in enumerate('xyz')]
    return types['sensor_msgs/msg/PointCloud2'](
        header, 1, len(points), fields, False, 12, points.nbytes, points.view(np.uint8).ravel(), True)


def export(inspection, result, output, pose_topic):
    """The pose topic must contain accepted map <- base_link poses only."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(exist_ok=True, parents=True)
    store = get_typestore(Stores.ROS2_HUMBLE)
    with reader(inspection) as source, reader(result) as bag, TemporaryDirectory(dir=output.parent) as tmp:
        scans = topic_connections(source, '/utlidar/cloud_base', 'sensor_msgs/msg/PointCloud2')
        poses = topic_connections(bag, pose_topic, 'geometry_msgs/msg/PoseStamped')
        if any(c.topic == '/current_scan' for c in bag.connections):
            raise ValueError('Result already contains /current_scan')
        messages = iter(source.messages(connections=scans))
        count = 0
        previous = -1
        stage = Path(tmp) / 'bag'
        with Writer(stage, version=8) as writer:
            copies = {c.id: writer.add_connection(
                c.topic, c.msgtype, typestore=bag.typestore, msgdef=c.msgdef.data or None, rihs01=c.digest or None,
                serialization_format=c.ext.serialization_format, offered_qos_profiles=c.ext.offered_qos_profiles)
                for c in bag.connections}
            layer = writer.add_connection('/current_scan', 'sensor_msgs/msg/PointCloud2', typestore=store)
            for c, t, raw in bag.messages():
                writer.write(copies[c.id], t, raw)
                if c not in poses:
                    continue
                pose = bag.deserialize(raw, c.msgtype)
                target = timestamp_ns(pose.header)
                if target <= previous:
                    raise ValueError('Accepted pose stamps must increase')
                previous = target
                for sc, _, data in messages:
                    scan = source.deserialize(data, sc.msgtype)
                    if timestamp_ns(scan.header) >= target:
                        break
                else:
                    raise ValueError(f'No scan for pose at {target}')
                cloud = transformed_cloud(scan, pose, store.types)
                writer.write(layer, t, store.serialize_cdr(cloud, layer.msgtype))
                count += 1
        if not count:
            raise ValueError('No accepted poses')
        stage.rename(output)
    return dict(transformed_scans=count)


def tracker_export(inspection, tracker_dir, output):
    """Build a diagnostic bag from localize3d's map.npz and trajectory.csv."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    store = get_typestore(Stores.ROS2_HUMBLE)
    types = store.types
    with (tracker_dir / 'trajectory.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    times = [float(row['t']) for row in rows]
    if not times or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Expected increasing tracker timestamps')
    with np.load(tracker_dir / 'map.npz') as data:
        points = data['xyz']
    output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=output.parent) as tmp, reader(inspection) as bag:
        base = Path(tmp) / 'poses'
        with Writer(base, version=8) as w:
            poses = w.add_connection('/accepted_pose', 'geometry_msgs/msg/PoseStamped', typestore=store)
            maps = w.add_connection('/map_points', 'sensor_msgs/msg/PointCloud2', typestore=store)
            scans = topic_connections(bag, '/utlidar/cloud_base', 'sensor_msgs/msg/PointCloud2')
            index = 0
            previous = -np.inf
            for c, _, raw in bag.messages(connections=scans):
                scan = bag.deserialize(raw, c.msgtype)
                t = stamp(scan)
                if t <= previous:
                    raise ValueError('Source scan stamps must be unique at CSV precision')
                previous = t
                if index == len(times) or t < times[index]:
                    continue
                if t != times[index]:
                    raise ValueError('Tracker row has no matching source scan')
                header = types['std_msgs/msg/Header'](scan.header.stamp, 'map')
                ns = timestamp_ns(header)
                if index == 0:
                    cloud = xyz_cloud(points, header, types)
                    w.write(maps, ns, store.serialize_cdr(cloud, maps.msgtype))
                row = rows[index]
                index += 1
                if row['status'] != 'matched':
                    continue
                xyz = types['geometry_msgs/msg/Point'](*(float(row[k]) for k in ('x', 'y', 'z')))
                quat = types['geometry_msgs/msg/Quaternion'](*(float(row[k]) for k in ('qx', 'qy', 'qz', 'qw')))
                pose = types[poses.msgtype](header, types['geometry_msgs/msg/Pose'](xyz, quat))
                w.write(poses, ns, store.serialize_cdr(pose, poses.msgtype))
            if index != len(times):
                raise ValueError('Tracker trajectory extends past the source recording')
        return export(inspection, base, output, '/accepted_pose')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inspection', type=Path)
    parser.add_argument('result', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--pose-topic', help='Accepted PoseStamped topic in a result bag; omit for a localize3d directory')
    args = parser.parse_args()
    try:
        stats = (export(**vars(args)) if args.pose_topic else
                 tracker_export(args.inspection, args.result, args.output))
        print(json.dumps(stats))
    except (OSError, ValueError) as exc:
        parser.exit(2, f'{exc}\n')
