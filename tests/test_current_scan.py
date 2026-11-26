import numpy as np
import pytest
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from mgts_nav.bag import cloud_xyz, reader
from mgts_nav.current_scan import export


@pytest.mark.parametrize('pose_stamp', [101, 102])
def test_exact_pose_and_spatial_rotation(tmp_path, pose_stamp):
    store = get_typestore(Stores.ROS2_HUMBLE)
    types = store.types

    def header(ns, frame):
        return types['std_msgs/msg/Header'](types['builtin_interfaces/msg/Time'](0, ns), frame)

    source, result, output = (tmp_path / name for name in ('source', 'result', 'output'))
    with Writer(source, version=8) as w:
        c = w.add_connection('/utlidar/cloud_base', 'sensor_msgs/msg/PointCloud2', typestore=store)
        fields = [types['sensor_msgs/msg/PointField'](s, i * 4, 7, 1) for i, s in enumerate('xyz')]
        points = np.array([[0, 1, 0], [0, 0, 0]], dtype='<f4')
        for ns in (100, 102, 103):
            msg = types[c.msgtype](header(ns, 'base_link'), 1, 2, fields, False, 12, 24,
                                   points.view(np.uint8).ravel(), True)
            w.write(c, ns, store.serialize_cdr(msg, c.msgtype))
    with Writer(result, version=8) as w:
        c = w.add_connection('/accepted', 'geometry_msgs/msg/PoseStamped', typestore=store)
        q = types['geometry_msgs/msg/Quaternion'](2**-0.5, 0., 0., 2**-0.5)
        p = types['geometry_msgs/msg/Point'](4., 5., 6.)
        msg = types[c.msgtype](header(pose_stamp, 'map'), types['geometry_msgs/msg/Pose'](p, q))
        w.write(c, 500, store.serialize_cdr(msg, c.msgtype))
    if pose_stamp == 101:
        with pytest.raises(ValueError, match='same header stamp'):
            export(source, result, output, '/accepted')
        assert not output.exists()
        return
    assert export(source, result, output, '/accepted') == {'transformed_scans': 1}
    with reader(output) as bag:
        assert {c.topic for c in bag.connections} == {'/accepted', '/current_scan'}
        c, t, raw = next(bag.messages(connections=[c for c in bag.connections if c.topic == '/current_scan']))
        cloud = bag.deserialize(raw, c.msgtype)
        assert t == 500 and cloud.header.frame_id == 'map'
        assert (cloud.header.stamp.sec, cloud.header.stamp.nanosec) == (0, 102)
        np.testing.assert_allclose(cloud_xyz(cloud), [[4., 5., 7.]])
