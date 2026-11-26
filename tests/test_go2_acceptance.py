"""Acceptance must detect loss even when bag format and log coverage are valid."""
import json
import subprocess
import sys

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from mgts_nav.ros_export import _pose


def test_go2_acceptance(tmp_path):
    store = get_typestore(Stores.ROS2_HUMBLE)
    types = store.types
    stamps = [1_795_600_000_123_456_789 + i * 100_000_000 for i in range(4)]

    def header(ns, frame):
        return types['std_msgs/msg/Header'](
            types['builtin_interfaces/msg/Time'](*divmod(ns, 10**9)), frame)

    def cloud(ns, frame):
        fields = [types['sensor_msgs/msg/PointField'](name, i * 4, 7, 1)
                  for i, name in enumerate('xyz')]
        xyz = np.array([[0, 0, 0], [1, 0, 1]], dtype='<f4')
        return types['sensor_msgs/msg/PointCloud2'](
            header(ns, frame), 1, 2, fields, False, 12, 24, xyz.view(np.uint8).ravel(), True)

    source = tmp_path / 'source'
    with Writer(source, version=8) as writer:
        conn = writer.add_connection('/utlidar/cloud_base', 'sensor_msgs/msg/PointCloud2', typestore=store)
        for ns in stamps:
            writer.write(conn, ns, store.serialize_cdr(cloud(ns, 'base_link'), conn.msgtype))
    for name, pose_stamps in [('full', stamps[1:]), ('truncated', stamps[1:2]),
                              ('shifted', [stamps[0], *stamps[2:]])]:
        with Writer(tmp_path / name, version=8) as writer:
            conn = writer.add_connection('/map_points', 'sensor_msgs/msg/PointCloud2', typestore=store)
            writer.write(conn, stamps[0], store.serialize_cdr(cloud(stamps[0], 'map'), conn.msgtype))
            conn = writer.add_connection('/pose', 'geometry_msgs/msg/PoseStamped', typestore=store)
            for ns in pose_stamps:
                pose = types[conn.msgtype](header(ns, 'map'), _pose(types, 0, 0, 0))
                writer.write(conn, ns, store.serialize_cdr(pose, conn.msgtype))

    timings = tmp_path / 'latency.csv'

    def log(statuses):
        timings.write_text('header_s,status,latency_ms,pending\n' + ''.join(
            f'{ns // 10**9 + (ns % 10**9) * 1e-9!r},{status},12,0\n'
            for ns, status in zip(stamps, statuses)), encoding='utf-8')

    def run(module, *args):
        return subprocess.run([sys.executable, '-m', f'mgts_nav.{module}', *map(str, args)],
                              capture_output=True, text=True)

    def accept(name, allowance=1):
        return run('result_bag_audit', source, tmp_path / name, '--timings', timings,
                   '--allow-missing-initial-scans', allowance)

    log(['missing_odometry', 'matched', 'prediction_only', 'matched'])
    assert accept('full').returncode == 0
    assert accept('full', 0).returncode == 2  # First scan needs an explicit allowance.
    assert accept('truncated').returncode == 2
    assert accept('shifted').returncode == 2  # Same count, different exact stamps.
    summary = run('result_bag_audit', source, tmp_path / 'truncated')
    assert summary.returncode == 0
    assert json.loads(summary.stdout)['poses'] == 1
    assert run('replay_audit', source, timings).returncode == 0
    log(['missing_odometry', 'matched', 'missing_odometry', 'matched'])
    assert accept('full').returncode == 2  # Extra loss in the log is also rejected.
    log(['missing_odometry'] * len(stamps))
    report = run('replay_audit', source, timings)
    assert report.returncode == 1
    assert json.loads(report.stdout)['pose_latency_ms'] is None
    assert accept('full').returncode == 2
