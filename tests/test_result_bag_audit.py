import numpy as np
import pytest
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from mgts_nav.result_bag_audit import audit
from mgts_nav.ros_export import _pose


@pytest.mark.parametrize("fault", [None, "foreign_stamp", "duplicate", "flat_map",
                                    "wrong_frame", "quaternion", "nan_position"])
def test_inspection_result(tmp_path, fault):
    store = get_typestore(Stores.ROS2_HUMBLE)
    types = store.types
    epoch = 1_795_600_000_000_000_000

    def header(ns, frame):
        t = types["builtin_interfaces/msg/Time"](*divmod(ns, 10**9))
        return types["std_msgs/msg/Header"](t, frame)

    def write(writer, connection, ns, msg):
        writer.write(connection, ns, store.serialize_cdr(msg, connection.msgtype))

    def cloud(ns, frame):
        fields = [types["sensor_msgs/msg/PointField"](name, i * 4, 7, 1)
                  for i, name in enumerate("xyz")]
        points = np.array([[0, 0, 0], [1, 0, float(fault != "flat_map")]], dtype="<f4")
        return types["sensor_msgs/msg/PointCloud2"](
            header(ns, frame), 1, 2, fields, False, 12, 24, points.view(np.uint8).ravel(), True)

    inspection, result = tmp_path / "input", tmp_path / "output"
    with Writer(inspection, version=8) as writer:
        scan = writer.add_connection("/scans", "sensor_msgs/msg/PointCloud2", typestore=store)
        for t in (epoch, epoch + 1, epoch + 2):
            write(writer, scan, t, cloud(t, "base_link"))
    with Writer(result, version=8) as writer:
        points = writer.add_connection("/xyz", "sensor_msgs/msg/PointCloud2", typestore=store)
        poses = writer.add_connection("/pose", "geometry_msgs/msg/PoseStamped", typestore=store)
        write(writer, points, epoch, cloud(epoch, "map"))
        times = [epoch, epoch + 2]
        if fault == "foreign_stamp":
            times[-1] += 1
        if fault == "duplicate":
            times[-1] = epoch
        for ns in times:
            pose = _pose(types, 0, 0, 0)
            if fault == "quaternion":
                pose.orientation.w = 0.1
            if fault == "nan_position":
                pose.position.x = np.nan
            msg = types["geometry_msgs/msg/PoseStamped"](
                header(ns, "odom" if fault == "wrong_frame" else "map"), pose)
            write(writer, poses, ns, msg)

    args = (inspection, result, "/scans", "/xyz", "/pose", "map")
    if fault:
        with pytest.raises(ValueError):
            audit(*args)
    else:
        assert audit(*args) == dict(map_messages=1, source_scans=3, poses=2,
                                   coverage=2/3, missing_scan_ns=[epoch + 1])
