"""Convert one ROS 2 LaserScan topic to PointCloud2, preserving other CDR messages."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from rosbags.rosbag2 import Writer, WriterError
from rosbags.typesys import Stores, get_typestore

from mgts_nav.bag import reader


def cloud(scan, types):
    increment = scan.angle_increment
    step = scan.time_increment
    lo, hi = scan.range_min, scan.range_max
    if (not np.isfinite([scan.angle_min, increment, step, lo, hi]).all()
            or increment == 0 or step < 0 or not 0 <= lo < hi):
        raise ValueError('Invalid LaserScan angle, time increment or range limits')
    ranges = np.asarray(scan.ranges, dtype=float)
    indices = np.flatnonzero(np.isfinite(ranges) & (ranges >= lo) & (ranges <= hi))
    angle = scan.angle_min + indices * increment
    points = np.column_stack((ranges[indices] * np.cos(angle), ranges[indices] * np.sin(angle),
                              np.zeros(len(indices)), indices * step)).astype('<f4')
    field = types['sensor_msgs/msg/PointField']
    fields = [field(name, i*4, 7, 1) for i, name in enumerate(('x', 'y', 'z', 'time'))]
    return types['sensor_msgs/msg/PointCloud2'](scan.header, 1, len(points), fields, False,
                                               16, points.nbytes, points.view(np.uint8).ravel(), True)


def convert(source, topic, output):
    if output.exists() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError('Select a new output directory outside the source recording')
    store = get_typestore(Stores.ROS2_HUMBLE)
    with reader(source) as bag:
        selected = [c for c in bag.connections if c.topic == topic]
        if not bag.is2 or len(selected) != 1 or selected[0].msgtype != 'sensor_msgs/msg/LaserScan':
            raise ValueError('Select one ROS 2 LaserScan connection by topic')
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent) as tmp:
            stage = Path(tmp) / 'converted'
            with Writer(stage, version=8) as writer:
                targets = {}
                for c in bag.connections:
                    qos = c.ext.offered_qos_profiles
                    if c == selected[0]:
                        target = writer.add_connection(topic, 'sensor_msgs/msg/PointCloud2',
                                                       typestore=store, offered_qos_profiles=qos)
                    else:
                        target = writer.add_connection(c.topic, c.msgtype, msgdef=c.msgdef.data,
                                                       rihs01=c.digest, serialization_format=c.ext.serialization_format,
                                                       offered_qos_profiles=qos)
                    targets[c.id] = target
                for c, t, raw in bag.messages():
                    target = targets[c.id]
                    if c == selected[0]:
                        msg = cloud(bag.deserialize(raw, c.msgtype), store.types)
                        raw = store.serialize_cdr(msg, target.msgtype)
                    writer.write(target, t, raw)
            stage.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('topic')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    try:
        convert(args.source, args.topic, args.output)
    except (OSError, ValueError, WriterError) as error:
        parser.exit(2, f'{error}\n')
