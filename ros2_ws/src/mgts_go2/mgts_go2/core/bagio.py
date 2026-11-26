"""Offline reading of the organizers' ROS 2 bags (MCAP or SQLite) with the ``rosbags`` library.

Only standard message types are deserialized, so the Unitree ``/lowstate``
definition is not needed here.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import cloud, se3
from .odom_buffer import OdomBuffer

ODOM_TOPIC = '/utlidar/robot_odom'
CLOUD_TOPIC = '/utlidar/cloud_base'


def find_bag_dir(path):
    """Accept the bag directory, its parent (the organizers' double nesting) or the .mcap file."""
    p = Path(path)
    if p.is_file():
        p = p.parent
    if (p / 'metadata.yaml').exists():
        return p
    found = sorted(p.rglob('metadata.yaml'))
    if len(found) == 1:
        return found[0].parent
    raise FileNotFoundError('expected exactly one metadata.yaml under %s, found %d' % (path, len(found)))


def stamp_of(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def odom_pose(msg):
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    return se3.from_pose((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))


@dataclass
class Scan:
    stamp: float
    log_time: float
    points: np.ndarray       # N x 3, frame of the message
    times: np.ndarray        # N, seconds after stamp (zeros if the field is absent)
    intensity: np.ndarray    # N
    frame_id: str


def iter_messages(bag, topics):
    """Yield (topic, log_time_s, message) in log order.

    Uses the pure-Python ``rosbags`` package when installed, otherwise the
    ROS 2 ``rosbag2_py`` API of the sourced ROS installation.
    """
    try:
        import rosbags  # noqa: F401
    except ImportError:
        yield from _iter_rosbag2_py(bag, topics)
        return
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
    store = get_typestore(Stores.ROS2_HUMBLE)
    with Reader(find_bag_dir(bag)) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        missing = set(topics) - {c.topic for c in conns}
        if missing:
            raise KeyError('topics not in bag %s: %s' % (bag, sorted(missing)))
        for conn, log_ns, raw in reader.messages(connections=conns):
            yield conn.topic, log_ns * 1e-9, store.deserialize_cdr(raw, conn.msgtype)


def _iter_rosbag2_py(bag, topics):
    import yaml
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    path = find_bag_dir(bag)
    meta = yaml.safe_load((path / 'metadata.yaml').read_text())
    storage = meta['rosbag2_bagfile_information'].get('storage_identifier', 'sqlite3')
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=storage),
                rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    missing = set(topics) - set(types)
    if missing:
        raise KeyError('topics not in bag %s: %s' % (bag, sorted(missing)))
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    classes = {t: get_message(types[t]) for t in topics}
    while reader.has_next():
        topic, raw, log_ns = reader.read_next()
        yield topic, log_ns * 1e-9, deserialize_message(raw, classes[topic])


def load(bag, cloud_topic=CLOUD_TOPIC, odom_topic=ODOM_TOPIC, every=1, limit=None):
    """Read all odometry and the selected scans. Returns (OdomBuffer, [Scan], odom_array).

    odom_array rows: stamp, x, y, z, qx, qy, qz, qw.
    """
    odom = OdomBuffer(horizon=1e12)
    rows = []
    scans = []
    k = 0
    for topic, log_t, msg in iter_messages(bag, [cloud_topic, odom_topic]):
        if topic == odom_topic:
            st = stamp_of(msg.header)
            if odom.add(st, odom_pose(msg)):
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                rows.append((st, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
            continue
        k += 1
        if (k - 1) % every:
            continue
        f = cloud.decode(msg)
        pts = cloud.xyz(f)
        n = len(pts)
        scans.append(Scan(stamp_of(msg.header), log_t, pts, f.get('time', np.zeros(n)),
                          f.get('intensity', np.zeros(n)), msg.header.frame_id))
        if limit and len(scans) >= limit:
            break
    return odom, scans, np.array(rows)
