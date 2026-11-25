import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from mgts_nav.bag import load_config, reader, scans, stamp
from mgts_nav.demo import PROFILE, generate
from mgts_nav.sync import audit


def test_unavailable_fields_in_recorded_imu(tmp_path):
    generate(tmp_path / "demo")
    source = tmp_path / "demo/inspection"
    target = tmp_path / "flagged"
    store = get_typestore(Stores.ROS2_HUMBLE)
    cfg = load_config(PROFILE)
    with reader(source) as bag, Writer(target, version=8) as writer:
        outputs = {c.id: writer.add_connection(c.topic, c.msgtype, typestore=store) for c in bag.connections}
        for c, t, raw in bag.messages():
            if c.topic == cfg["imu"]:
                m = bag.deserialize(raw, c.msgtype)
                elapsed = stamp(m) - 250.
                m.angular_velocity_covariance = np.zeros(9)
                m.linear_acceleration_covariance = np.zeros(9)
                if 2 <= elapsed < 3:
                    m.angular_velocity_covariance[0] = -1
                    m.angular_velocity.z = 8.
                if 6 <= elapsed < 7:
                    m.linear_acceleration_covariance[0] = -1
                    m.linear_acceleration.z = -20.
                raw = store.serialize_cdr(m, c.msgtype)
            writer.write(outputs[c.id], t, raw)
    original = list(scans(source, cfg))
    result = list(scans(target, cfg))
    missing = [scan.t for scan in result[1:] if scan.gyro_delta is None]
    np.testing.assert_allclose(missing, 252.035 + np.arange(11) / 10, rtol=0, atol=1e-8)
    for scan, ref in zip(result, original, strict=True):
        if 252.15 < scan.t < 253.:
            assert scan.gyro_delta is None, (scan.t, scan.gyro_delta)
        elif scan.t < 251.9 or scan.t > 253.2:
            assert scan.gyro_delta == ref.gyro_delta
        np.testing.assert_array_equal(scan.xy, ref.xy)
    report = audit(target, cfg)
    assert report["coverage"]["imu"]["covered"] == 99
    assert report["imu_measurements"]["linear_acceleration_available"] == 451
    leveled = list(scans(target, dict(cfg, gravity_level=True)))
    for scan in leveled:
        if 252.15 < scan.t < 253. or 256.15 < scan.t < 257.:
            assert scan.gyro_delta is None and scan.empty_reason == "leveling_unavailable"
    assert leveled[-1].gyro_delta is not None and len(leveled[-1].xy) > 0
    assert audit(target, dict(cfg, gravity_level=True))["imu_measurements"]["gyro_delta"]["available"] == sum(
        scan.gyro_delta is not None for scan in leveled)
