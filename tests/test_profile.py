import json
import os
import subprocess
import sys

import numpy as np
import pytest
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from mgts_nav.profile import draft_profile


def test_redirected_profile_preserves_unicode(tmp_path):
    bag = tmp_path / 'bag'
    with Writer(bag, version=8) as writer:
        writer.add_connection('/lidar', 'sensor_msgs/msg/PointCloud2',
                              typestore=get_typestore(Stores.ROS2_HUMBLE))
    command = [sys.executable, '-m', 'mgts_nav.profile', str(bag), '--base', '基座']
    env = dict(os.environ, PYTHONIOENCODING='cp1251')
    stdout = subprocess.check_output(command, env=env)
    assert json.loads(stdout.decode('ascii'))['profile']['base_frame'] == '基座'
    target = tmp_path / 'draft.json'
    subprocess.run([*command, '--out', target], env=env, check=True)
    content = target.read_text(encoding='utf-8')
    assert '基座' in content and json.loads(content) == json.loads(stdout)


def test_ambiguous_lidars_and_recorded_static_calibration(tmp_path):
    store = get_typestore(Stores.ROS2_HUMBLE)
    types = store.types
    header = types['std_msgs/msg/Header']
    stamp = types['builtin_interfaces/msg/Time'](1, 0)
    cloud_type = 'sensor_msgs/msg/PointCloud2'
    tf_type = 'tf2_msgs/msg/TFMessage'
    with Writer(tmp_path / 'bag', version=8) as writer:
        for key in ('front', 'rear'):
            conn = writer.add_connection('/'+key, cloud_type, typestore=store)
            msg = types[cloud_type](header(stamp, key), 1, 0, [], False, 0, 0,
                                    np.empty(0, np.uint8), True)
            writer.write(conn, 1_000_000_000, store.serialize_cdr(msg, cloud_type))
        trans = types['geometry_msgs/msg/Transform'](
            types['geometry_msgs/msg/Vector3'](.25, 0., .3),
            types['geometry_msgs/msg/Quaternion'](0., 0., 0., 1.))
        edge = types['geometry_msgs/msg/TransformStamped'](header(stamp, 'robot'), 'front', trans)
        conn = writer.add_connection('/tf_static', tf_type, typestore=store)
        writer.write(conn, 1_000_000_000, store.serialize_cdr(types[tf_type]([edge]), tf_type))
    ambiguous = draft_profile(tmp_path / 'bag', base='robot')
    assert ambiguous['status'] == 'draft' and ambiguous['profile']['lidar'] is None
    assert ambiguous['candidates']['lidar'] == ['/front', '/rear']
    assert any('--lidar' in question for question in ambiguous['questions'])
    selected = draft_profile(tmp_path / 'bag', lidar='/front', base='robot')['profile']
    assert selected['lidar_frame'] == 'front'
    assert selected['base_from_lidar_xyz_m'] == pytest.approx([.25, 0., .3])
    assert selected['base_from_imu_xyzw'] is None and selected['stationary_s'] is None
    rear = draft_profile(tmp_path / 'bag', lidar='/rear', base='robot')['profile']
    assert rear['base_from_lidar_xyz_m'] is None
    with pytest.raises(ValueError, match='candidates'):
        draft_profile(tmp_path / 'bag', lidar='/missing')


@pytest.mark.parametrize('change', ['lidar', 'child', 'stable'])
def test_full_frame_verification_within_one_connection(tmp_path, change):
    store = get_typestore(Stores.ROS2_HUMBLE)
    types = store.types
    header = types['std_msgs/msg/Header']
    stamp = types['builtin_interfaces/msg/Time'](1, 0)
    cloud_type = 'sensor_msgs/msg/PointCloud2'
    odom_type = 'nav_msgs/msg/Odometry'
    vector = types['geometry_msgs/msg/Vector3'](0., 0., 0.)
    pose = types['geometry_msgs/msg/Pose'](types['geometry_msgs/msg/Point'](0., 0., 0.),
                                           types['geometry_msgs/msg/Quaternion'](0., 0., 0., 1.))
    twist = types['geometry_msgs/msg/Twist'](vector, vector)
    with Writer(tmp_path / 'bag', version=8) as writer:
        cloud_conn = writer.add_connection('/lidar', cloud_type, typestore=store)
        odom_conn = writer.add_connection('/kinematic', odom_type, typestore=store)
        for i in range(2):
            cloud_frame = 'second_sensor' if i and change == 'lidar' else 'body'
            body = 'other_body' if i and change == 'child' else 'body'
            cloud = types[cloud_type](header(stamp, cloud_frame), 1, 0, [], False, 0, 0,
                                      np.empty(0, np.uint8), True)
            odom = types[odom_type](header(stamp, 'world'), body,
                                    types['geometry_msgs/msg/PoseWithCovariance'](pose, np.zeros(36)),
                                    types['geometry_msgs/msg/TwistWithCovariance'](twist, np.zeros(36)))
            writer.write(cloud_conn, i+1, store.serialize_cdr(cloud, cloud_type))
            writer.write(odom_conn, i+1, store.serialize_cdr(odom, odom_type))
    quick = draft_profile(tmp_path / 'bag')
    assert quick['frame_check'] == 'first_per_connection'
    assert quick['observed_frames']['/kinematic']['checked_messages'] == 1
    assert quick['profile']['base_from_lidar_xyz_m'] == [0, 0, 0]
    result = draft_profile(tmp_path / 'bag', base='body', verify_frames=True)
    config = result['profile']
    assert result['frame_check'] == 'all_messages'
    assert result['observed_frames']['/kinematic']['checked_messages'] == 2
    if change == 'stable':
        assert config == quick['profile']
    elif change == 'child':
        assert config['base_frame'] is None and config['base_from_lidar_xyzw'] is None
        assert result['observed_frames']['/kinematic']['child_frame_ids'] == ['body', 'other_body']
    else:
        assert config['lidar_frame'] is None and config['base_from_lidar_xyz_m'] is None
        assert result['observed_frames']['/lidar']['header_frame_ids'] == ['body', 'second_sensor']

@pytest.mark.parametrize('ros1,topics', [(False, ['/scan']), (True, ['/rear', '/front'])])
def test_profile_guides_laser_scan_conversion(tmp_path, ros1, topics):
    from rosbags.rosbag1 import Writer as Ros1Writer

    bag = tmp_path / ('original.bag' if ros1 else 'original')
    store = get_typestore(Stores.ROS1_NOETIC if ros1 else Stores.ROS2_HUMBLE)
    with (Ros1Writer(bag) if ros1 else Writer(bag, version=8)) as writer:
        for topic in topics:
            writer.add_connection(topic, 'sensor_msgs/msg/LaserScan', typestore=store)
    cmd = [sys.executable, '-m', 'mgts_nav.profile', str(bag)]
    report = json.loads(subprocess.check_output(cmd))
    assert report['profile']['lidar'] is None and report['candidates']['lidar'] == []
    hint, = [line for line in report['questions'] if 'tools.laser_scan' in line]
    assert all(topic in hint for topic in topics)
    assert ('rosbags.convert' in hint) == ros1
    assert ('docs/ros1-validation.md' if ros1 else 'docs/intake.md') in hint
    explicit = subprocess.run([*cmd, '--lidar', topics[0]], capture_output=True, text=True)
    assert explicit.returncode == 2 and hint in explicit.stderr


def test_profile_prefers_existing_cloud_with_laser_scan(tmp_path):
    store = get_typestore(Stores.ROS2_HUMBLE)
    with Writer(tmp_path/'mixed', version=8) as writer:
        writer.add_connection('/scan', 'sensor_msgs/msg/LaserScan', typestore=store)
        cloud = writer.add_connection('/points', 'sensor_msgs/msg/PointCloud2', typestore=store)
        header = store.types['std_msgs/msg/Header'](store.types['builtin_interfaces/msg/Time'](1, 0), 'front')
        msg = store.types[cloud.msgtype](header, 1, 0, [], False, 0, 0, np.empty(0, np.uint8), True)
        writer.write(cloud, 1_000_000_000, store.serialize_cdr(msg, cloud.msgtype))
    report = draft_profile(tmp_path/'mixed')
    assert report['profile']['lidar'] == '/points' and report['profile']['lidar_frame'] == 'front'
    assert report['candidates']['lidar'] == ['/points']
    assert all('tools.laser_scan' not in line for line in report['questions'])
