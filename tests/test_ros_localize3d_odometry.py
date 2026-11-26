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
import pytest

from mgts_nav import bag, localize3d


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
                           cloud_frame='base_link', odom_frame='odom',
                           odom_child_frame='base_link', wrong_odom_frame=False,
                           get_logger=lambda: SimpleNamespace(warning=warnings.append))
    def process():
        if len(node.odometry) >= 2:
            processed.append(ros.Motion(*zip(*node.odometry)).at(10.05))

    node.process = process

    def deliver(t, x):
        position = SimpleNamespace(x=x, y=0., z=0.)
        orientation = SimpleNamespace(x=0., y=0., z=0., w=1.)
        message = SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=10, nanosec=t), frame_id='odom'),
            child_frame_id='base_link',
            pose=SimpleNamespace(pose=SimpleNamespace(position=position, orientation=orientation)))
        ros.Localizer.on_odom(node, message)
        return message

    first = deliver(0, 0.)
    middle = deliver(50_000_000, 1.)
    original = list(node.odometry)
    for t in (50_000_000, 25_000_000, -10_000_000_000):  # duplicate, late, clock reset
        deliver(t, 999.)
        assert list(node.odometry) == original
    assert len(processed) == 1
    last = deliver(100_000_000, 2.)
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

    # Frame changes must be rejected even after valid data have been accepted.
    cloud = SimpleNamespace(header=SimpleNamespace(
        stamp=SimpleNamespace(sec=10, nanosec=50_000_000), frame_id='base_link'))
    odom_conn = SimpleNamespace(topic='/utlidar/robot_odom', msgtype='odom')
    cloud_conn = SimpleNamespace(topic='/utlidar/cloud_base', msgtype='cloud')

    class Recording:
        connections = [odom_conn, cloud_conn]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def messages(self, connections):
            messages = [first, middle, last] if connections == [odom_conn] else [cloud]
            return [(connections[0], 0, msg) for msg in messages]

        def deserialize(self, raw, msgtype):
            return raw

    monkeypatch.setattr(bag, 'reader', lambda _: Recording())
    monkeypatch.setattr(bag, 'cloud_xyz', lambda _: np.array([[1., 0., 0.]]))
    monkeypatch.setattr(bag, 'cloud_field', lambda *args: np.array([0.]))
    baseline = list(localize3d.scans('unused'))
    assert len(baseline) == 1
    np.testing.assert_array_equal(baseline[0][1], [[1., 0., 0.]])
    np.testing.assert_array_equal(baseline[0][2].xyz, [1., 0., 0.])

    node.pending.clear()
    node.log_scan = lambda scan, status: ros.Localizer.log_scan(node, scan, status)
    for target, attribute, expected, topic in [
        (cloud.header, 'frame_id', 'base_link', cloud_conn.topic),
        (middle.header, 'frame_id', 'odom', odom_conn.topic),
        (middle, 'child_frame_id', 'base_link', odom_conn.topic),
    ]:
        for actual in ('wrong', ''):
            setattr(target, attribute, actual)
            with pytest.raises(ValueError) as error:
                list(localize3d.scans('unused'))
            detail = str(error.value)
            assert all(text in detail for text in (topic, '10.050000000',
                                                   repr(expected), repr(actual)))
            previous_processed = len(processed)
            if topic == odom_conn.topic:
                node.pending.append((10.05, None, None, perf_counter()))
                ros.Localizer.on_odom(node, middle)
                assert not node.odometry and not node.pending
                assert node.wrong_odom_frame
            ros.Localizer.on_cloud(node, cloud)
            assert len(processed) == previous_processed
            assert 'wrong_frame' in node.timings.getvalue()
            assert detail in warnings
        setattr(target, attribute, expected)
        node.wrong_odom_frame = False
    restored = list(localize3d.scans('unused'))
    assert len(restored) == len(baseline)
    np.testing.assert_array_equal(restored[0][1], baseline[0][1])
    np.testing.assert_array_equal(restored[0][2].xyz, baseline[0][2].xyz)
    with pytest.raises(ValueError, match='explicit sensor transform'):
        list(localize3d.scans('unused', cloud_frame='lidar'))
