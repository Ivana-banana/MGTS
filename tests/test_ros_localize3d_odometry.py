"""Exercise the ROS callback with import stubs; no ROS installation required."""

import csv
import importlib.util
import io
import sys
from collections import deque
from pathlib import Path
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np


def test_odometry_rejects_non_increasing_stamps_and_resumes(monkeypatch):
    imports = {
        'rclpy': (), 'rclpy.node': ('Node',),
        'rclpy.qos': ('DurabilityPolicy', 'QoSProfile', 'qos_profile_sensor_data'),
        'rclpy.utilities': ('remove_ros_args',),
        'geometry_msgs.msg': ('PoseStamped', 'TransformStamped'),
        'nav_msgs.msg': ('Odometry',), 'sensor_msgs.msg': ('PointCloud2',),
        'sensor_msgs_py.point_cloud2': ('create_cloud_xyz32', 'read_points'),
        'std_msgs.msg': ('Header',), 'tf2_ros': ('TransformBroadcaster',),
    }
    for name in imports:
        parent = name.split('.')[0]
        monkeypatch.setitem(sys.modules, parent, ModuleType(parent))
    for name, attributes in imports.items():
        module = ModuleType(name)
        for attribute in attributes:
            setattr(module, attribute, object)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        'mgts_nav._ros_localize3d_issue221',
        Path(__file__).resolve().parents[1] / 'mgts_nav' / 'ros_localize3d.py')
    ros = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ros)
    warnings, processed = [], []
    node = SimpleNamespace(odometry=deque(maxlen=3), pending=deque([object()]),
                           get_logger=lambda: SimpleNamespace(warning=warnings.append))
    def process():
        if len(node.odometry) >= 2:
            processed.append(ros.Motion(*zip(*node.odometry)).at(10.05))

    node.process = process

    def deliver(t, x):
        position = SimpleNamespace(x=x, y=0., z=0.)
        orientation = SimpleNamespace(x=0., y=0., z=0., w=1.)
        message = SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=10, nanosec=t)),
            pose=SimpleNamespace(pose=SimpleNamespace(position=position, orientation=orientation)))
        ros.Localizer.on_odom(node, message)

    deliver(0, 0.)
    deliver(50_000_000, 1.)
    original = list(node.odometry)
    for t in (50_000_000, 25_000_000, -10_000_000_000):  # duplicate, late, clock reset
        deliver(t, 999.)
        assert list(node.odometry) == original
    assert len(processed) == 1
    deliver(100_000_000, 2.)
    assert [sample[0] for sample in node.odometry] == [10., 10.05, 10.1]
    assert len(processed) == 2
    np.testing.assert_array_equal(processed[-1].xyz, [1., 0., 0.])
    np.testing.assert_array_equal(processed[-1].rotation.as_quat(), [0., 0., 0., 1.])
    assert len(warnings) == 3
    assert 'duplicate' in warnings[0]
    assert all('backward' in warning for warning in warnings[1:])
    assert 'Restart node after a clock reset' in warnings[-1]

    node.timings = io.StringIO()
    node.csv = csv.writer(node.timings)
    ros.Localizer.log_scan(node, (10., None, None, perf_counter()), 'missing_odometry')
    assert 'missing_odometry' in node.timings.getvalue()
    assert 'Skipping scan 10.000000000: missing_odometry' == warnings[-1]
