"""Add /current_scan to a ROS 2 result bag using exact, accepted tracker poses."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

from .bag import cloud_xyz, reader
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
    fields = [types['sensor_msgs/msg/PointField'](name, i * 4, 7, 1) for i, name in enumerate('xyz')]
    return types['sensor_msgs/msg/PointCloud2'](
        pose.header, 1, len(points), fields, False, 12, points.nbytes, points.view(np.uint8).ravel(), True)


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
                c.topic, c.msgtype, msgdef=c.msgdef.data, rihs01=c.digest,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inspection', type=Path)
    parser.add_argument('result', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--pose-topic', required=True, help='PoseStamped containing accepted registrations only')
    args = parser.parse_args()
    try:
        print(json.dumps(export(**vars(args))))
    except (OSError, ValueError) as exc:
        parser.exit(2, f'{exc}\n')
