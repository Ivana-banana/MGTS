import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml
from PIL import Image
from rosbags.highlevel import AnyReader
from rosbags.interfaces import QosDurability, QosHistory, QosReliability

from mgts_nav.ros_export import export


@pytest.fixture
def inputs(tmp_path):
    pixels = np.array([[0, 0, 0, 0, 0], [0, 205, 0, 0, 0], [254, 0, 0, 0, 0]], dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path / 'map.pgm')
    map_yaml = tmp_path / 'map.yaml'
    map_yaml.write_text(yaml.safe_dump(dict(image='map.pgm', resolution=.25, origin=[-2.5, 3., 0.],
                                            negate=0, occupied_thresh=.65, free_thresh=.196)))
    trajectory = tmp_path / 'trajectory.csv'
    trajectory.write_text('t,x,y,yaw,status\n1700000000.123456789,2,-3,0,matched\n'
                          '1700000000.478901234,-2,7,1.5707963267948966,matched\n')
    return map_yaml, trajectory


def test_cli_round_trip_geometry_stamps_and_qos(inputs, tmp_path):
    output = tmp_path / 'result'
    completed = subprocess.run([sys.executable, '-m', 'mgts_nav.ros_export', *map(str, inputs),
                                str(output), '--frame', 'site'], check=True, capture_output=True, text=True)
    first, last = 1700000000123456789, 1700000000478901234
    assert json.loads(completed.stdout)['start_ns'] == first
    with AnyReader([output]) as reader:
        topics = {c.topic: c for c in reader.connections}
        assert {k: v.msgcount for k, v in topics.items()} == {'/map': 1, '/pose': 2, '/path': 1}
        for topic in ['/map', '/path']:
            assert topics[topic].ext.offered_qos_profiles[0].durability == QosDurability.TRANSIENT_LOCAL
        assert topics['/pose'].ext.offered_qos_profiles[0].durability == QosDurability.VOLATILE
        displays = yaml.safe_load((Path(__file__).resolve().parents[1] /
                                   'configs/inspection.rviz').read_text(encoding='utf-8'))
        subscriptions = {d['Topic']['Value']: d['Topic']
                         for d in displays['Visualization Manager']['Displays']}
        for topic, connection in topics.items():
            profile, = connection.ext.offered_qos_profiles
            assert profile.reliability == QosReliability.RELIABLE
            assert profile.history == QosHistory.KEEP_LAST
            assert profile.depth == (10 if topic == '/pose' else 1)
            subscription = subscriptions[topic]
            assert subscription['Reliability Policy'] == 'Reliable'
            assert subscription['Durability Policy'] == ('Volatile' if topic == '/pose' else 'Transient Local')
        msgs = {topic: [] for topic in topics}
        for conn, t, raw in reader.messages():
            msg = reader.deserialize(raw, conn.msgtype)
            assert msg.header.frame_id == 'site'
            assert msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec == t
            msgs[conn.topic].append((t, msg))
    ns, grid = msgs['/map'][0]
    assert ns == first and (grid.info.width, grid.info.height) == (5, 3)
    assert grid.info.resolution == .25 and grid.info.origin.position.x == -2.5
    assert grid.info.origin.position.y == 3 and grid.info.origin.orientation.w == 1
    np.testing.assert_array_equal(grid.data.reshape(3, 5),
                                  [[0, 100, 100, 100, 100], [100, -1, 100, 100, 100], [100] * 5])
    assert [t for t, _ in msgs['/pose']] == [first, last]
    ns, path = msgs['/path'][0]
    assert ns == last and len(path.poses) == 2
    for (t, live), stamped in zip(msgs['/pose'], path.poses):
        assert live == stamped
    p = path.poses[-1].pose
    assert (p.position.x, p.position.y, p.position.z) == (-2, 7, 0)
    assert 2 * math.atan2(p.orientation.z, p.orientation.w) == pytest.approx(math.pi / 2)


def test_invalid_trajectory_never_creates_bag(inputs, tmp_path):
    map_yaml, trajectory = inputs
    trajectory.write_text('t,x,y,yaw\n42.00000000001,0,0,0\n42.00000000002,0,0,0\n')
    target = tmp_path / 'failed'
    with pytest.raises(ValueError, match='non-increasing'):
        export(map_yaml, trajectory, target)
    assert not target.exists()


def test_existing_output_left_intact(inputs, tmp_path):
    target = tmp_path / 'existing'
    target.mkdir()
    sentinel = target / 'sentinel'
    sentinel.write_text('keep')
    with pytest.raises(FileExistsError):
        export(*inputs, target)
    assert sentinel.read_text() == 'keep'
