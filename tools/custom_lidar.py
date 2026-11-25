"""Convert the documented demo_lidar CDR topic to a standard ROS 2 PointCloud2 bag."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from rosbags.rosbag2 import Reader, Writer
from rosbags.typesys import Stores, get_types_from_msg, get_typestore


PACKET = 'demo_lidar/msg/Packet'
CLOUD = 'sensor_msgs/msg/PointCloud2'


def load_schema(directory):
    store = get_typestore(Stores.ROS2_HUMBLE)
    definitions = {}
    for name in ('Return', 'Packet'):
        path = directory / (name + '.msg')
        if not path.is_file():
            raise ValueError(f'Missing external schema for demo_lidar/msg/{name}: {path}')
        definitions.update(get_types_from_msg(path.read_text(encoding='utf-8'), f'demo_lidar/msg/{name}'))
    store.register(definitions)
    return store


def pointcloud(packet, types):
    if packet.unit != 'mm':
        raise ValueError(f'Unsupported distance unit {packet.unit!r}, expected mm')
    if packet.header.frame_id != 'lidar_flu':
        raise ValueError(f'Unknown lidar frame {packet.header.frame_id!r}, expected lidar_flu (forward/left/up)')
    xyz = np.asarray([(p.forward, p.left, p.up) for p in packet.returns], dtype=np.float64)
    xyz = (xyz.reshape(-1, 3) / 1000).astype('<f4')
    field = types['sensor_msgs/msg/PointField']
    fields = [field(axis, i * 4, 7, 1) for i, axis in enumerate('xyz')]
    return types[CLOUD](packet.header, 1, len(xyz), fields, False, 12, xyz.nbytes,
                        xyz.view(np.uint8).ravel(), True)


def convert(bag, topic, schema, output):
    if output.exists():
        raise FileExistsError(f'Choose a new destination: {output}')
    store = load_schema(schema)
    count = 0
    with Reader(bag) as reader:
        selected = [c for c in reader.connections if c.topic == topic]
        if not selected:
            raise ValueError(f'Missing topic {topic!r}. Available: {[c.topic for c in reader.connections]}')
        for connection in selected:
            if connection.msgtype != PACKET:
                raise ValueError(f'{topic}: expected {PACKET}, found {connection.msgtype}')
            if connection.ext.serialization_format != 'cdr':
                raise ValueError(f'{topic}: expected CDR serialization')
            if connection.digest != store.hash_rihs01(PACKET):
                raise ValueError(f'{topic}: recorded type hash differs from external schema or is unavailable')
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent) as tmp:
            stage = Path(tmp) / 'converted'
            with Writer(stage, version=9) as writer:
                standard = writer.add_connection('/points', CLOUD, typestore=store)
                for _, timestamp, raw in reader.messages(connections=selected):
                    cloud = pointcloud(store.deserialize_cdr(raw, PACKET), store.types)
                    writer.write(standard, timestamp, store.serialize_cdr(cloud, CLOUD))
                    count += 1
            if not count:
                raise ValueError(f'No packets on {topic}')
            stage.rename(output)
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('topic')
    parser.add_argument('schema', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    try:
        print(f'{convert(args.bag, args.topic, args.schema, args.output)} clouds written to /points')
    except (OSError, ValueError) as error:
        parser.exit(2, f'{error}\n')
