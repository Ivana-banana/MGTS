import struct
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from mgts_nav import bag


def field(name, offset, datatype):
    return SimpleNamespace(name=name, offset=offset, datatype=datatype, count=1)


def organized_cloud(big_endian=False):
    # Coordinates have different widths, noncanonical order and padding after every row.
    byteorder = '>' if big_endian else '<'
    data = bytearray([0xA5] * 100)
    points = [(1, 2, 3), (4, 5, 6), (-1, -2, -3), (0.25, 0.5, 0.75)]
    for i, (x, y, z) in enumerate(points):
        offset = (i // 2) * 50 + (i % 2) * 24
        struct.pack_into(byteorder + 'd', data, offset, z)
        struct.pack_into(byteorder + 'f', data, offset + 12, x)
        struct.pack_into(byteorder + 'f', data, offset + 16, y)
    return SimpleNamespace(height=2, width=2, fields=[field('z', 0, 8), field('x', 12, 7),
                                                       field('y', 16, 7), field('intensity', 20, 7)],
                           is_bigendian=big_endian, point_step=24, row_step=50, data=data)


@pytest.mark.parametrize('big_endian', [True, False])
def test_organized_cloud_mixed_fields_and_padding(big_endian):
    np.testing.assert_array_equal(bag.cloud_xyz(organized_cloud(big_endian)),
                                  [[1, 2, 3], [4, 5, 6], [-1, -2, -3], [.25, .5, .75]])


@pytest.mark.parametrize('problem', ['missing_z', 'integer_x', 'short_row', 'field_outside_point'])
def test_invalid_cloud_layout(problem):
    msg = organized_cloud()
    if problem == 'missing_z':
        msg.fields.pop(0)
    if problem == 'integer_x':
        msg.fields[1].datatype = 6
    if problem == 'short_row':
        msg.row_step = 47
    if problem == 'field_outside_point':
        msg.fields[2].offset = 23
    with pytest.raises(ValueError):
        bag.cloud_xyz(msg)


def test_interpolation_window_requires_brackets_and_contiguous_times():
    times = np.array([0., .1, .2, .8, .9])
    window = bag.interval_samples(times, .025, .175, .3)
    np.testing.assert_array_equal(times[window], [0., .1, .2])
    assert bag.interval_samples(times, .1, .8, .3) is None
    assert bag.interval_samples(times, .5, .5, .3) is None
    assert bag.interval_samples(times, -.1, .1, .3) is None
    assert bag.interval_samples(times, .8, 1., .3) is None
    np.testing.assert_array_equal(times[bag.interval_samples(times, .8, .8, .3)], [.8])


def header(ms, frame):
    return SimpleNamespace(stamp=SimpleNamespace(sec=10, nanosec=ms * 1_000_000), frame_id=frame)


def odom(ms, yaw):
    q = SimpleNamespace(x=0., y=0., z=np.sin(yaw / 2), w=np.cos(yaw / 2))
    pose = SimpleNamespace(position=SimpleNamespace(x=ms / 1000, y=0.), orientation=q)
    return SimpleNamespace(header=header(ms, 'odom'), child_frame_id='base_link',
                           pose=SimpleNamespace(pose=pose))


def test_scan_fusion_uses_header_clock_handles_wrap_and_sensor_gaps(monkeypatch):
    clouds = []
    for ms in [50, 150, 200, 400, 600]:
        msg = organized_cloud()
        msg.header = header(ms, 'lidar')
        clouds.append(msg)
    # +179 and -179 degrees interpolate through pi, not zero.
    odometry = [odom(ms, np.radians(yaw)) for ms, yaw in
                [(0, 177), (100, 179), (200, -179), (600, -175)]]
    gyro = [SimpleNamespace(header=header(ms, 'imu'), angular_velocity_covariance=np.zeros(9),
                            angular_velocity=SimpleNamespace(x=0., y=0., z=rate))
            for ms, rate in [(0, .1), (100, .1), (200, 2.1), (600, 2.1)]]
    streams = {'/cloud': ('sensor_msgs/msg/PointCloud2', clouds),
               '/imu': ('sensor_msgs/msg/Imu', gyro), '/odom': ('nav_msgs/msg/Odometry', odometry)}

    class MemoryBag:
        connections = [SimpleNamespace(topic=topic, msgtype=item[0]) for topic, item in streams.items()]

        def messages(self, connections):
            for conn in connections:
                # All arrival times are identical. Only message header times carry motion.
                for msg in streams[conn.topic][1]:
                    yield conn, 99_000_000_000, msg

        def deserialize(self, raw, _):
            return raw

    monkeypatch.setattr(bag, 'reader', lambda _: nullcontext(MemoryBag()))
    cfg = dict(lidar='/cloud', imu='/imu', odom='/odom', lidar_frame='lidar', imu_frame='imu',
               base_frame='base_link', odom_frame='odom', base_from_lidar_xyz_m=[.2, 0., .3],
               base_from_lidar_xyzw=[0., 0., 0., 1.], base_from_imu_xyzw=[0., 0., 0., 1.],
               height_band_m=[.9, 1.2], stationary_s=.1, max_sync_gap_s=.25)
    scans = list(bag.scans('in-memory', cfg))
    np.testing.assert_allclose(scans[1].xy, [[.45, .5]])
    np.testing.assert_allclose(scans[1].origin, [.2, 0.])
    np.testing.assert_allclose(scans[1].odom, [.15, 0., np.pi])
    assert scans[0].gyro_delta is None
    assert scans[1].gyro_delta == pytest.approx(.025)
    assert scans[2].gyro_delta == pytest.approx(.075)
    assert scans[3].odom is None and scans[3].gyro_delta is None
    assert scans[4].odom is not None and scans[4].gyro_delta is None

    cfg['imu'] = None
    del cfg['base_from_imu_xyzw']
    without_imu = list(bag.scans('in-memory', cfg))
    assert len(without_imu) == len(scans)
    assert all(scan.gyro_delta is None for scan in without_imu)
    np.testing.assert_allclose(without_imu[1].odom, scans[1].odom)
