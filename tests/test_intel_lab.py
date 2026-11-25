import numpy as np
from tools.intel_lab import points


def test_front_laser_direction_and_range_cutoff():
    ranges = np.full(180, 81.83)
    ranges[[0, 30, 45, 80, 90, 179]] = [2, np.nan, 20, -1, 3, 4]
    expected = [[0, -2, 0], [3, 0, 0], [4*np.sin(np.deg2rad(1)), 4*np.cos(np.deg2rad(1)), 0]]
    np.testing.assert_allclose(points(ranges), expected, atol=1e-6)


def test_odometry_uses_its_own_ipc_clock_and_keeps_boundary_samples(tmp_path):
    import gzip
    from mgts_nav.bag import reader
    from tools.intel_lab import odom_rows, write_window

    log = tmp_path / "test.gz"
    with gzip.open(log, "wt") as stream:
        for i in [3, 0, 4, 2, 1]:
            stream.write(f"ODOM {i} 2 0.6 0.7 -0.3 0 {10+i} host 99\n")
    rows = [(11_500_000_000, ["FLASER", "180"] + ["1"]*180),
            (12_500_000_000, ["FLASER", "180"] + ["1"]*180)]
    path = tmp_path / "bag"
    write_window(path, rows, odom_rows(log))
    with reader(path) as bag:
        conns = [c for c in bag.connections if c.topic == "/odom"]
        times = []
        for c, ns, raw in bag.messages(connections=conns):
            m = bag.deserialize(raw, c.msgtype)
            assert m.header.frame_id == "odom" and m.child_frame_id == "base"
            assert ns == m.header.stamp.sec*10**9 + m.header.stamp.nanosec
            assert m.pose.pose.position.x == ns/1e9 - 10
            np.testing.assert_allclose(m.pose.pose.orientation.z, np.sin(.3))
            assert m.twist.twist.linear.x == .7 and m.twist.twist.angular.z == -.3
            times.append(ns)
        assert times == [11*10**9, 12*10**9, 13*10**9]
