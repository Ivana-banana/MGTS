import pytest
from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore

from mgts_nav.bag import reader
from tools.drop_odometry import copy_window


def messages(path):
    with reader(path) as bag:
        return [(c.topic, t, bytes(raw)) for c, t, raw in bag.messages()]


@pytest.mark.parametrize("storage", [StoragePlugin.MCAP, StoragePlugin.SQLITE3])
def test_only_odometry_in_half_open_delivery_interval_is_removed(tmp_path, storage):
    store = get_typestore(Stores.ROS2_HUMBLE)
    # Header stamps deliberately differ from delivery time and run backwards.
    header = store.types['std_msgs/msg/Header']
    time = store.types['builtin_interfaces/msg/Time']
    source = tmp_path / 'source'
    with Writer(source, version=8, storage_plugin=storage) as writer:
        conns = [writer.add_connection(topic, 'std_msgs/msg/Header', typestore=store)
                 for topic in ['/utlidar/robot_odom', '/cloud', '/other']]
        for i in range(10):
            for j, c in enumerate(conns):
                raw = store.serialize_cdr(header(time(100 - i, j), str(i)), c.msgtype)
                writer.write(c, 1000 + i * 10, raw)
    original = messages(source)
    control, dropped = tmp_path / 'control', tmp_path / 'dropped'
    copy_window(source, control, 1010, 1090)
    result = copy_window(source, dropped, 1010, 1090, 1030, 1060)
    expected = [m for m in original if 1010 <= m[1] < 1090]
    assert messages(control) == expected
    assert messages(dropped) == [m for m in expected if not (
        m[0] == '/utlidar/robot_odom' and 1030 <= m[1] < 1060)]
    assert result['removed'] == {'/utlidar/robot_odom': 3}
    assert result['counts']['/cloud'] == 8
    with pytest.raises(FileExistsError):
        copy_window(source, control, 1010, 1090)
    with pytest.raises(ValueError):
        copy_window(source, tmp_path / 'bad', 1010, 1090, 1030)
