import json
import subprocess
import sys

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from mgts_nav.bag import cloud_xyz, reader
from tools.laser_scan import convert


def test_convert_scan_timing_and_map(tmp_path):
    store = get_typestore(Stores.ROS2_HUMBLE)
    types = store.types
    src, dst = tmp_path / 'laser', tmp_path / 'cloud'
    ranges = np.full(180, 3., dtype=np.float32)
    ranges[:5] = [np.nan, 1., .5, np.inf, 4.]
    with Writer(src, version=8) as writer:
        lidar = writer.add_connection('/scan', 'sensor_msgs/msg/LaserScan', typestore=store)
        other = writer.add_connection('/note', 'std_msgs/msg/String', typestore=store)
        note = bytes(store.serialize_cdr(types[other.msgtype]('office'), other.msgtype))
        writer.write(other, 99, note)
        for i in range(10):
            stamp = types['builtin_interfaces/msg/Time'](5, i*100000000)
            header = types['std_msgs/msg/Header'](stamp, 'sensor')
            msg = types[lidar.msgtype](header, 0., -1.79, -.01, .0005, .1, 1., 4.,
                                       ranges, np.empty(0, dtype=np.float32))
            writer.write(lidar, 100+i, store.serialize_cdr(msg, lidar.msgtype))
    convert(src, '/scan', dst)
    with reader(dst) as bag:
        messages = list(bag.messages())
        assert len(messages) == 11 and messages[0][1:] == (99, note)
        c, t, raw = messages[1]
        msg = bag.deserialize(raw, c.msgtype)
        assert (t, msg.header.stamp.sec, msg.header.frame_id) == (100, 5, 'sensor')
        xyz = cloud_xyz(msg)
        assert len(xyz) == 177
        np.testing.assert_allclose(xyz[0], [np.cos(.01), -np.sin(.01), 0], atol=1e-6)
        values = np.frombuffer(msg.data, '<f4').reshape(-1, 4)
        np.testing.assert_allclose(values[:2, 3], [.0005, .002])
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps(dict(lidar='/scan', lidar_frame='sensor', odom=None, imu=None,
                                       base_from_lidar_xyz_m=[0, 0, 0], base_from_lidar_xyzw=[0, 0, 0, 1],
                                       height_band_m=[-.1, .1], stationary_s=0, max_sync_gap_s=.5)))
    output = tmp_path / 'map'
    subprocess.run([sys.executable, '-m', 'mgts_nav', 'map', dst, '--config', profile, '--out', output],
                   check=True)
    assert json.loads((output / 'summary.json').read_bytes())['scans'] == 10

def write_office_scans(source, output):
    """Place planar office returns in the generator's original angular bins."""
    from mgts_nav.demo import ANGLES

    store = get_typestore(Stores.ROS2_HUMBLE)
    step = 2 * np.pi / len(ANGLES)
    scan_type = 'sensor_msgs/msg/LaserScan'
    with reader(source) as bag, Writer(output, version=8) as writer:
        targets = {c.id: writer.add_connection(
            c.topic, scan_type if c.msgtype.endswith('/PointCloud2') else c.msgtype, typestore=store)
            for c in bag.connections}
        for c, t, raw in bag.messages():
            target = targets[c.id]
            if target.msgtype == scan_type:
                msg = bag.deserialize(raw, c.msgtype)
                x, y = cloud_xyz(msg)[:, :2].T
                bins = np.rint((np.arctan2(y, x) - ANGLES[0]) / step).astype(int) % len(ANGLES)
                ranges = np.full(len(ANGLES), np.inf, dtype=np.float32)
                ranges[bins] = np.hypot(x, y)
                scan = store.types[scan_type](msg.header, ANGLES[0], ANGLES[-1], step, .0002, .1,
                                              .05, 20., ranges, np.empty(0, dtype=np.float32))
                raw = store.serialize_cdr(scan, scan_type)
            writer.write(target, t, raw)


def test_ros1_laser_scan_through_solve(tmp_path):
    import hashlib
    from itertools import zip_longest

    from mgts_nav.demo import PROFILE, generate

    def cli(*args):
        subprocess.run([sys.executable, '-X', 'utf8', '-m', *map(str, args)], check=True)

    office = tmp_path / 'office'
    generate(office, seed=41)
    inputs = [[], []]
    for role in ('mapping', 'inspection'):
        reference = tmp_path / f'{role}-ros2'
        original = tmp_path / f'{role}.bag'
        restored = tmp_path / f'{role}-converted'
        write_office_scans(office / role, reference)
        cli('rosbags.convert', '--src', reference, '--dst', original, '--dst-typestore', 'ros1_noetic')
        original_bytes = original.read_bytes()
        assert original_bytes.startswith(b'#ROSBAG V2.0\n')
        cli('rosbags.convert', '--src', original, '--dst', restored, '--dst-typestore', 'ros2_humble')
        # All CDR fields and both clocks must survive the round trip, including skipped beams.
        with reader(reference) as before, reader(restored) as after, reader(original) as ros1:
            for a, b, c in zip_longest(before.messages(), after.messages(), ros1.messages()):
                assert a is not None and b is not None and c is not None
                assert a[0].topic == b[0].topic == c[0].topic
                assert a[1] == b[1] == c[1] and a[2] == b[2]
                if a[0].msgtype.endswith('/LaserScan'):
                    scan = before.deserialize(a[2], a[0].msgtype)
                    old = ros1.deserialize(c[2], c[0].msgtype)
                    assert old.header.stamp == scan.header.stamp and old.header.frame_id == scan.header.frame_id
                    np.testing.assert_array_equal(old.ranges, scan.ranges)
                    assert old.time_increment == scan.time_increment
        for index, src in enumerate((reference, restored)):
            dst = tmp_path / f'{role}-cloud-{index}'
            cli('tools.laser_scan', src, '/demo/cloud', dst)
            inputs[index].append(dst)
        assert original.read_bytes() == original_bytes

    products = []
    for index, bags in enumerate(inputs):
        output = tmp_path / f'solve-{index}'
        cli('mgts_nav.solve', *bags, '--config', PROFILE, '--out', output)
        assert json.loads((output/'inspection/summary.json').read_bytes())['status'] == {'matched': 100}
        products.append(output)
    for name in ('map/map.pgm', 'map/map.yaml', 'map/trajectory.csv', 'inspection/trajectory.csv',
                 'inspection/summary.json', 'inspection/continuous.csv', 'baseline/trajectory.csv'):
        left, right = [(out/name).read_bytes() for out in products]
        assert left == right, name
        print(name, hashlib.sha256(left).hexdigest())
