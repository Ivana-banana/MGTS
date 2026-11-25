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
