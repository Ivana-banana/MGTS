import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

from mgts_nav import metrics
from mgts_nav.export2d import HEIGHT_BAND_M, Row, build_grid, read_bag, read_csv, write_trajectory, yaw
from mgts_nav.localize3d import Pose


def room_scans():
    # Walls of a 6 x 4 m room, a floor patch below the band and a ceiling above it.
    # Walls run through cell centres (0.025 m offset) so each lies inside one 5 cm cell column.
    edge = np.linspace(0, 1, 600)
    walls = 0.025 + np.concatenate([np.column_stack([6 * edge, 0 * edge]), np.column_stack([6 * edge, 0 * edge + 4]),
                                    np.column_stack([0 * edge, 4 * edge]), np.column_stack([0 * edge + 6, 4 * edge])])
    for x, y, heading in [(2, 2, 0.3), (3, 1.5, 1.9), (4, 2.5, -2.4)]:
        pose = Pose(Rotation.from_euler('z', heading), np.array([x, y, 0.3]))
        world = np.concatenate([np.column_stack([walls, np.full(len(walls), 0.8)]),
                                [[3, 2, 0.0], [3, 2.1, 2.5]]])  # floor and ceiling at the centre
        yield pose.rotation.inv().apply(world - pose.xyz), pose


def test_room_walls_occupied_centre_free_outside_unknown(tmp_path):
    grid, used, total = build_grid(room_scans(), 0.05, HEIGHT_BAND_M)
    grid.save(tmp_path)
    meta = yaml.safe_load((tmp_path / 'map.yaml').read_text())
    pixels = np.array(Image.open(tmp_path / meta['image']))

    def cell(x, y):
        col, row = np.floor((np.array([x, y]) - meta['origin'][:2]) / meta['resolution']).astype(int)
        assert 0 <= row < len(pixels) and 0 <= col < pixels.shape[1]
        return pixels[len(pixels) - 1 - row, col]

    assert (used, total) == (3, 3)
    assert [cell(3.025, 0.025), cell(0.025, 2.025), cell(6.025, 1.025), cell(3.025, 4.025)] == [0] * 4
    assert cell(3.02, 2.02) == 254 and cell(1.5, 3.0) == 254
    assert cell(-0.075, 2.025) == 205 and cell(3.025, 4.125) == 205  # behind the walls


def test_default_band_matches_official_config():
    config = json.loads((Path(__file__).parents[1] / 'configs/go2-official.json').read_text())
    assert list(HEIGHT_BAND_M) == config['height_band_m']


def test_yaw_comes_from_quaternion_with_pitch_and_roll():
    quat = Rotation.from_euler('ZYX', [2.5, 0.2, -0.1]).as_quat()
    assert abs(yaw(quat) - 2.5) < 1e-12


def test_tum_and_csv_round_trip_consistently(tmp_path):
    # 0.1 s cadence with nanosecond digits; the stamp text must survive exactly.
    rows = [Row(f'1795605722.{i * 100_000_000 + 7:09d}', 1795605722 + i * 0.1 + 7e-9,
                (0.1 * i, -0.2 * i, 0.3), tuple(map(float, Rotation.from_euler("ZYX", [0.4 * i - 1, 0.1, 0.05]).as_quat())),
                'matched') for i in range(1, 6)]
    write_trajectory(rows, tmp_path)
    lines = (tmp_path / 'trajectory.tum').read_text().splitlines()
    assert [line.split()[0] for line in lines] == [row.stamp for row in rows]
    tum = np.array([[float(v) for v in line.split()] for line in lines])
    assert tum.shape == (5, 8)
    assert np.array_equal(tum[:, 1:4], [row.xyz for row in rows])
    assert np.array_equal(tum[:, 4:], [row.quat for row in rows])
    planar = metrics.read_csv(tmp_path / 'trajectory.csv')
    assert [p.t for p in planar] == list(tum[:, 0])
    assert np.allclose([(p.x, p.y, p.yaw) for p in planar],
                       np.column_stack([tum[:, 1:3], Rotation.from_quat(tum[:, 4:]).as_euler('ZYX')[:, 0]]),
                       atol=1e-12)
    assert np.allclose([p.yaw for p in planar], [0.4 * i - 1 for i in range(1, 6)])
    assert metrics.evaluate(planar, planar)['ate_xy_rmse_m'] == 0
    source = tmp_path / 'source.csv'
    source.write_text('t,x,y,z,qx,qy,qz,qw,status\n' + ''.join(
        f'{r.stamp},{",".join(map(repr, (*r.xyz, *r.quat)))},{r.status}\n' for r in rows))
    again, frame = read_csv(source)
    assert again == rows and frame == 'map'


def test_reads_pose3d_from_rosbag2(tmp_path):
    store = get_typestore(Stores.ROS2_HUMBLE)
    t = store.types
    quat = Rotation.from_euler('z', 1.2).as_quat()
    bag = tmp_path / 'result'
    with Writer(bag, version=8) as writer:
        conn = writer.add_connection('/mgts/pose3d', 'geometry_msgs/msg/PoseStamped', typestore=store)
        for i in range(3):
            header = t['std_msgs/msg/Header'](t['builtin_interfaces/msg/Time'](1795605722, 7 + i * 100_000_000), 'map')
            pose = t['geometry_msgs/msg/Pose'](t['geometry_msgs/msg/Point'](1. + i, 2., 0.3),
                                               t['geometry_msgs/msg/Quaternion'](*quat))
            message = t['geometry_msgs/msg/PoseStamped'](header, pose)
            writer.write(conn, 10 + i, store.serialize_cdr(message, conn.msgtype))
    rows, frame = read_bag(bag)
    assert frame == 'map' and len(rows) == 3
    assert [row.stamp for row in rows] == ['1795605722.000000007', '1795605722.100000007',
                                           '1795605722.200000007']
    assert rows[2].xyz == (3., 2., 0.3) and abs(rows[0].pose[2] - 1.2) < 1e-12
