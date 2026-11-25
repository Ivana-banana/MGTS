import json
import subprocess
import sys

import pytest
from rosbags.convert import convert
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from mgts_nav.bag import load_config, reader, scans, stamp
from mgts_nav.demo import PROFILE, generate
from mgts_nav.sync import audit


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("imu-sync")
    generate(root/"data")
    return root/"data/inspection"


@pytest.mark.parametrize("availability,expected", [("all", 99), ("partial", 88), ("none", 0)])
def test_sync_measurements_match_navigation(demo, tmp_path, availability, expected):
    cfg = load_config(PROFILE)
    ros2 = tmp_path/"inspection"
    store = get_typestore(Stores.ROS2_HUMBLE)
    with reader(demo) as bag, Writer(ros2, version=8) as writer:
        outputs = {c.id: writer.add_connection(c.topic, c.msgtype, typestore=store) for c in bag.connections}
        for c, t, raw in bag.messages():
            if c.topic == cfg["imu"]:
                msg = bag.deserialize(raw, c.msgtype)
                elapsed = stamp(msg)-250.
                if availability == "none" or (availability == "partial" and 2 <= elapsed < 3):
                    msg.angular_velocity_covariance = msg.angular_velocity_covariance.copy()
                    msg.angular_velocity_covariance[0] = -1
                raw = store.serialize_cdr(msg, c.msgtype)
            writer.write(outputs[c.id], t, raw)
    ros1 = tmp_path/"inspection.bag"
    convert([ros2], ros1, dst_storage="sqlite3", dst_version=8, compress=None, compress_mode="file",
            default_typestore=store, typestore=get_typestore(Stores.ROS1_NOETIC), exclude_topics=(),
            include_topics=(), exclude_msgtypes=(), include_msgtypes=())
    for path in (ros2, ros1):
        output = subprocess.check_output([sys.executable, "-m", "mgts_nav.sync", path, "--config", PROFILE,
                                          "--require-complete"], text=True)
        report = json.loads(output)
        assert report["coverage"]["imu"]["covered"] == 99
        measurement = report["imu_measurements"]
        assert measurement["angular_velocity_available"] == {"all": 501, "partial": 451, "none": 0}[availability]
        assert measurement["linear_acceleration_available"] == measurement["messages"] == 501
        assert measurement["gyro_delta"]["available"] == expected
        for profile in (cfg, dict(cfg, gravity_level=True)):
            measured = audit(path, profile)["imu_measurements"]["gyro_delta"]
            stream = list(scans(path, profile))
            intervals = [(a.t, b.t) for a, b in zip(stream, stream[1:]) if b.gyro_delta is None]
            runs = measured["missing_runs"]
            assert measured["available"] == 99-len(intervals)
            assert sum(run["intervals"] for run in runs) == len(intervals)
            for run in runs:
                subset = [(a, b) for a, b in intervals if run["start_s"] <= a < b <= run["end_s"]]
                assert len(subset) == run["intervals"]
                assert subset[0][0] == run["start_s"] and subset[-1][1] == run["end_s"]
