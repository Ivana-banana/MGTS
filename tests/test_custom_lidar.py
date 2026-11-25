from pathlib import Path
import shutil

import numpy as np
import pytest
from rosbags.rosbag2 import Reader, Writer

from mgts_nav.bag import cloud_xyz, inspect_bag
from tools.custom_lidar import CLOUD, PACKET, convert, load_schema


SCHEMA = Path(__file__).resolve().parents[1] / 'configs' / 'custom_lidar'


def write_packet(path, frame='lidar_flu', unit='mm'):
    store = load_schema(SCHEMA)
    types = store.types
    header = types['std_msgs/msg/Header'](types['builtin_interfaces/msg/Time'](4, 17), frame)
    point = types['demo_lidar/msg/Return']
    packet = types[PACKET](header, unit, [point(1250, -250, 500), point(-3000, 0, 2000)])
    with Writer(path, version=9) as writer:
        connection = writer.add_connection('/raw', PACKET, typestore=store)
        writer.write(connection, 4_002_000_017, store.serialize_cdr(packet, PACKET))
    return store, header


def test_packet_matches_standard_cloud(tmp_path):
    source, output = tmp_path / 'bag', tmp_path / 'standard'
    store, header = write_packet(source)
    assert inspect_bag(source, inventory=True) == [dict(topic='/raw', type=PACKET, messages=1)]
    assert convert(source, '/raw', SCHEMA, output) == 1
    expected = np.array([[1.25, -.25, .5], [-3., 0., 2.]], dtype='<f4')
    field = store.types['sensor_msgs/msg/PointField']
    standard = store.types[CLOUD](header, 1, 2, [field('x', 0, 7, 1), field('y', 4, 7, 1),
                                  field('z', 8, 7, 1)], False, 12, 24, expected.view('u1').ravel(), True)
    with Reader(output) as reader:
        (conn, record_time, raw), = reader.messages()
        assert (conn.topic, conn.msgtype, record_time) == ('/points', CLOUD, 4_002_000_017)
        assert raw == store.serialize_cdr(standard, CLOUD)
        np.testing.assert_array_equal(cloud_xyz(store.deserialize_cdr(raw, CLOUD)), expected)
    with pytest.raises(ValueError, match='found sensor_msgs/msg/PointCloud2'):
        convert(output, '/points', SCHEMA, tmp_path / 'wrong')


@pytest.mark.parametrize(('frame', 'unit', 'error'), [
    ('lidar_flu', 'ticks', 'Unsupported distance unit'), ('optical', 'mm', 'Unknown lidar frame')])
def test_unknown_contract(tmp_path, frame, unit, error):
    write_packet(tmp_path / 'bag', frame, unit)
    with pytest.raises(ValueError, match=error):
        convert(tmp_path / 'bag', '/raw', SCHEMA, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_external_schema(tmp_path):
    write_packet(tmp_path / 'bag')
    schema = tmp_path / 'schema'
    schema.mkdir()
    shutil.copy(SCHEMA / 'Packet.msg', schema)
    with pytest.raises(ValueError, match='Missing external schema for demo_lidar/msg/Return'):
        convert(tmp_path / 'bag', '/raw', schema, tmp_path / 'out')
    (schema / 'Return.msg').write_text('float32 forward\nint32 left\nint32 up\n', encoding='utf-8')
    with pytest.raises(ValueError, match='recorded type hash differs'):
        convert(tmp_path / 'bag', '/raw', schema, tmp_path / 'out')
