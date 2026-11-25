"""Generate five "unfamiliar" recordings for an intake rehearsal of the final.

Each scenario differs from mgts_nav.demo in one family of conventions:

a  MCAP, Ouster/robot-style topic names, frames other than base_link, /tf_static
   with a chained calibration (base_footprint <- base_link <- os_sensor <- os_lidar/os_imu)
   and a dynamic /tf; the inspection is delivered as a bare .mcap file.
b  One recording: mapping, a stop, transit, a stop, then the inspection.
c  Organised PointCloud2 with x,y,z float64, intensity, t (uint32 ns from header), ring,
   NaN for missing returns; the lidar is pitched down; calibration only in a YAML file;
   points are measured along the sweep, so the scan is distorted by motion.
d  No IMU; odometry with covariances and dropouts; header clock is the robot boot clock
   (far from record time) and odometry headers run 0.25 s ahead of the lidar.
e  Single-plane lidar pitched 6 deg down, so it sees the floor beyond about 3.3 m;
   the calibration is given in words only. Used to check self_check on a flat scan.

Layout: OUT/<scenario>/data is what a team receives (bags + ORGANIZERS.md);
OUT/<scenario>/truth holds the reference trajectories and the expected profile.

    python tools/foreign_bags.py out/foreign
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mgts_nav.demo import WALLS, cross, pose_at  # noqa: E402

STORE = get_typestore(Stores.ROS2_HUMBLE)
T = STORE.types
CLOUD, IMU, ODOM, TF = ('sensor_msgs/msg/PointCloud2', 'sensor_msgs/msg/Imu',
                        'nav_msgs/msg/Odometry', 'tf2_msgs/msg/TFMessage')
PF = {'f4': 7, 'f8': 8, 'u4': 6, 'u2': 4, 'u1': 2}
CEILING = 2.7


def stamp(ns):
    return T['builtin_interfaces/msg/Time'](*divmod(int(ns), 1_000_000_000))


def header(ns, frame):
    return T['std_msgs/msg/Header'](stamp(ns), frame)


def vec(x, y, z):
    return T['geometry_msgs/msg/Vector3'](float(x), float(y), float(z))


def quat(q):
    return T['geometry_msgs/msg/Quaternion'](*map(float, q))


def transform(ns, parent, child, xyz, q):
    return T['geometry_msgs/msg/TransformStamped'](
        header(ns, parent), child, T['geometry_msgs/msg/Transform'](vec(*xyz), quat(q)))


def wall_distance(origin, directions):
    """Horizontal distance from origin along unit directions to the static walls."""
    edge = WALLS[:, 1] - WALLS[:, 0]
    relative = WALLS[:, 0] - origin
    denom = cross(directions[:, None], edge)
    valid = np.abs(denom) > 1e-10
    distance = np.divide(cross(relative, edge), denom, where=valid, out=np.zeros_like(denom))
    along = np.divide(cross(relative, directions[:, None]), denom, where=valid, out=np.zeros_like(denom))
    return np.where(valid & (distance > 0) & (along >= 0) & (along <= 1), distance, np.inf).min(axis=1)


class Lidar:
    """Multi-ring lidar: rays in the sensor frame, cast against walls, floor and ceiling."""

    def __init__(self, base_from_lidar_xyz, base_from_lidar_xyzw, rings, columns=240):
        self.offset = np.asarray(base_from_lidar_xyz, float)
        self.rot = Rotation.from_quat(base_from_lidar_xyzw)
        elevation = np.deg2rad(np.asarray(rings, float))[:, None]
        azimuth = np.linspace(-np.pi, np.pi, columns, endpoint=False)[None, :]
        self.shape = (len(rings), columns)
        self.dirs = np.stack([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth),
                              np.sin(elevation) * np.ones_like(azimuth)], -1).reshape(-1, 3)
        self.column = np.tile(np.arange(columns), len(rings))

    def ranges(self, poses):
        """poses: one planar pose per ray (N,3) in the world. Returns 3D ranges."""
        d_base = self.rot.apply(self.dirs)
        c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        origin = poses[:, :2] + np.column_stack([c * self.offset[0] - s * self.offset[1],
                                                 s * self.offset[0] + c * self.offset[1]])
        horiz = np.linalg.norm(d_base[:, :2], axis=1)
        dxy = d_base[:, :2] / np.maximum(horiz, 1e-12)[:, None]
        world_dir = np.column_stack([c * dxy[:, 0] - s * dxy[:, 1], s * dxy[:, 0] + c * dxy[:, 1]])
        if np.allclose(origin, origin[0]):
            hit = wall_distance(origin[0], world_dir)
        else:
            # During a sweep each ray has its own origin; group rays that share one.
            hit = np.empty(len(d_base))
            _, groups = np.unique(origin, axis=0, return_inverse=True)
            for g in np.unique(groups):
                mask = groups.ravel() == g
                hit[mask] = wall_distance(origin[mask][0], world_dir[mask])
        r = np.where(horiz > 1e-6, hit / np.maximum(horiz, 1e-12), np.inf)
        dz = d_base[:, 2]
        floor = np.where(dz < -1e-9, -self.offset[2] / np.where(dz < -1e-9, dz, -1), np.inf)
        ceil = np.where(dz > 1e-9, (CEILING - self.offset[2]) / np.where(dz > 1e-9, dz, 1), np.inf)
        r = np.minimum(r, np.minimum(floor, ceil))
        r[r > 40] = np.inf
        return r


def cloud_msg(ns, frame, columns):
    """columns: list of (name, dtype, array) with equal shapes (H, W)."""
    height, width = columns[0][2].shape
    fields, offset, names = [], 0, []
    for name, dtype, _ in columns:
        size = np.dtype(dtype).itemsize
        offset = -(-offset // size) * size
        fields.append((name, offset, dtype))
        offset += size
    step = -(-offset // 8) * 8
    record = np.zeros((height, width), np.dtype({'names': [f[0] for f in fields],
                                                 'formats': ['<' + f[2] for f in fields],
                                                 'offsets': [f[1] for f in fields], 'itemsize': step}))
    for name, _, array in columns:
        record[name] = array
    pf = [T['sensor_msgs/msg/PointField'](name, off, PF[dtype], 1) for name, off, dtype in fields]
    return T[CLOUD](header(ns, frame), height, width, pf, False, step, step * width,
                    record.view(np.uint8).ravel(), False)


def imu_msg(ns, frame, base_rate_z, base_from_imu_xyzw, rng):
    inv = Rotation.from_quat(base_from_imu_xyzw).inv()
    gyro = inv.apply([0., 0., base_rate_z + 0.015]) + rng.normal(0, 0.002, 3)
    acc = inv.apply([0., 0., 9.81]) + rng.normal(0, 0.02, 3)
    return T[IMU](header(ns, frame), quat([0, 0, 0, 1]), np.r_[-1.0, np.zeros(8)], vec(*gyro),
                  np.eye(3).ravel() * 4e-6, vec(*acc), np.eye(3).ravel() * 4e-4)


def odom_msg(ns, frame, child, local, yaw, twist=(0., 0.), covariance=0.0):
    cov = np.diag([covariance, covariance, 1e6, 1e6, 1e6, covariance * 4]).ravel()
    pose = T['geometry_msgs/msg/Pose'](T['geometry_msgs/msg/Point'](float(local[0]), float(local[1]), 0.),
                                       quat([0, 0, np.sin(yaw / 2), np.cos(yaw / 2)]))
    tw = T['geometry_msgs/msg/Twist'](vec(twist[0], 0, 0), vec(0, 0, twist[1]))
    return T[ODOM](header(ns, frame), child, T['geometry_msgs/msg/PoseWithCovariance'](pose, cov),
                   T['geometry_msgs/msg/TwistWithCovariance'](tw, cov))


def rate(traj, e):
    return (traj(e + 1e-3)[2] - traj(e - 1e-3)[2]) / 2e-3


def drifted(traj, e, start, active):
    """Per-run odometry frame with 4.5 % scale and heading drift, as in the demo."""
    pose = traj(e)
    c, s = np.cos(-start[2]), np.sin(-start[2])
    d = pose[:2] - start[:2]
    local = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
    a = 0.004 * active
    local = 1.045 * np.array([np.cos(a) * local[0] - np.sin(a) * local[1],
                              np.sin(a) * local[0] + np.cos(a) * local[1]])
    return local, pose[2] - start[2] + 0.007 * active


class Recording:
    def __init__(self, path, storage='sqlite3'):
        plugin = StoragePlugin.MCAP if storage == 'mcap' else StoragePlugin.SQLITE3
        self.writer = Writer(path, version=8, storage_plugin=plugin)
        self.conns, self.queue = {}, []

    def add(self, record_ns, topic, typ, msg):
        self.queue.append((int(record_ns), topic, typ, msg))

    def close(self):
        with self.writer as w:
            for record_ns, topic, typ, msg in sorted(self.queue, key=lambda q: (q[0], q[1])):
                if topic not in self.conns:
                    self.conns[topic] = w.add_connection(topic, typ, typestore=STORE)
                w.write(self.conns[topic], record_ns, STORE.serialize_cdr(msg, typ))


def write_truth(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        out = csv.writer(f)
        out.writerow(['t', 'x', 'y', 'yaw'])
        for t, pose in rows:
            yaw = np.angle(np.exp(1j * pose[2]))
            out.writerow([f'{t:.9f}', pose[0], pose[1], yaw])


def ring_points(lidar, pose, rng, dropout=0.1):
    r = lidar.ranges(np.repeat(pose[None], len(lidar.dirs), 0))
    r = r + rng.normal(0, 0.012, r.shape)
    r[rng.random(r.shape) < dropout] = np.inf
    return r


# --------------------------------------------------------------------- scenario a
def scenario_a(root, seed):
    data, truth = root / 'data', root / 'truth'
    data.mkdir(parents=True)
    truth.mkdir()
    q_sensor = Rotation.from_euler('z', 90, degrees=True)
    q_lidar = Rotation.from_euler('z', 180, degrees=True)
    chain = [('base_footprint', 'base_link', [0, 0, 0.12], [0, 0, 0, 1]),
             ('base_link', 'os_sensor', [0.15, 0.0, 0.45], q_sensor.as_quat()),
             ('os_sensor', 'os_lidar', [0, 0, 0.03618], q_lidar.as_quat()),
             ('os_sensor', 'os_imu', [0.006253, -0.011775, 0.007645], [0, 0, 0, 1])]
    lidar_xyz = [0.15, 0, 0.12 + 0.45 + 0.03618]
    lidar_q = (q_sensor * q_lidar).as_quat()
    imu_q = q_sensor.as_quat()
    lidar = Lidar(lidar_xyz, lidar_q, [-1.0, 0.0, 1.0])
    expected = dict(lidar='/ouster/points', imu='/imu/data', odom='/odom', lidar_frame='os_lidar',
                    imu_frame='os_imu', base_frame='base_footprint', odom_frame='odom',
                    base_from_lidar_xyz_m=lidar_xyz, base_from_lidar_xyzw=lidar_q.tolist(),
                    base_from_imu_xyzw=imu_q.tolist(), height_band_m=[0.15, 1.5],
                    max_sync_gap_s=0.35, stationary_s=0.5)
    wall0 = 1_764_150_000_000_000_000
    for i, run in enumerate(('mapping', 'inspection')):
        rng = np.random.default_rng(seed + i)
        dur = 20.0 if run == 'mapping' else 10.0
        start0 = wall0 + i * 600_000_000_000
        traj = lambda e, run=run: pose_at(run, e)  # noqa: E731
        start = traj(0.0)
        rec = Recording(data / run, 'mcap')
        for parent, child, xyz, q in chain:
            rec.add(start0, '/tf_static', TF, T[TF]([transform(start0, parent, child, xyz, q)]))
        rows = []
        for k in range(int(dur * 100) + 1):  # IMU 100 Hz
            e = k / 100
            rec.add(start0 + k * 10_000_000 + 1_000_000, '/imu/data', IMU,
                    imu_msg(start0 + k * 10_000_000, 'os_imu', rate(traj, e), imu_q, rng))
        for k in range(int(dur * 20) + 1):  # odom 20 Hz + dynamic TF
            e = k / 20
            ns = start0 + k * 50_000_000
            local, yaw = drifted(traj, e, start, np.clip(e - 1, 0, dur - 2))
            rec.add(ns + 3_000_000, '/odom', ODOM, odom_msg(ns, 'odom', 'base_footprint', local, yaw))
            rec.add(ns + 3_000_000, '/tf', TF, T[TF]([transform(ns, 'odom', 'base_footprint',
                                                                [*local, 0], [0, 0, np.sin(yaw / 2), np.cos(yaw / 2)])]))
        for k in range(int((dur - 0.05) * 10) + 1):  # lidar 10 Hz
            e = 0.05 + k / 10
            ns = start0 + int(round(e * 1e9))
            pose = traj(e)
            r = ring_points(lidar, pose, rng).reshape(lidar.shape)
            xyz = (lidar.dirs.reshape(*lidar.shape, 3) * np.where(np.isfinite(r), r, 0)[..., None])
            xyz[~np.isfinite(r)] = np.nan
            rec.add(ns + 12_000_000, '/ouster/points', CLOUD, cloud_msg(ns, 'os_lidar', [
                ('x', 'f4', xyz[..., 0]), ('y', 'f4', xyz[..., 1]), ('z', 'f4', xyz[..., 2]),
                ('intensity', 'f4', np.full(lidar.shape, 100.0))]))
            rows.append((ns / 1e9, pose))
        rec.close()
        write_truth(truth / f'{run}_truth.csv', rows)
    # Deliver the inspection as a bare single-file MCAP, as organisers often do.
    single = next((data / 'inspection').glob('*.mcap'))
    single.rename(data / 'inspection_run2.mcap')
    shutil.rmtree(data / 'inspection')
    (data / 'ORGANIZERS.md').write_text(
        'Две записи робота с лидаром Ouster в формате MCAP: `mapping/` — обход для построения карты, '
        '`inspection_run2.mcap` — инспекция.\nКалибровка датчиков записана в /tf_static. '
        'В начале каждой записи робот стоит около 1 с.\n', encoding='utf-8')
    (truth / 'expected_profile.json').write_text(json.dumps(expected, indent=2), encoding='utf-8')


# --------------------------------------------------------------------- scenario b
def combined_pose(e):
    """mapping 0-20 s, stop to 24 s, transit to 28 s, stop to 30 s, inspection 30-40 s."""
    if e <= 20:
        return pose_at('mapping', e)
    end = pose_at('mapping', 20.0)
    start = pose_at('inspection', 0.0) + np.array([0, 0, 2 * np.pi])
    if e <= 24:
        return end
    if e <= 28:
        u = (e - 24) / 4
        u = u * u * (3 - 2 * u)
        return end + u * (start - end)
    if e <= 30:
        return start
    return pose_at('inspection', e - 30) + np.array([0, 0, 2 * np.pi])


def scenario_b(root, seed):
    data, truth = root / 'data', root / 'truth'
    data.mkdir(parents=True)
    truth.mkdir()
    profile = json.loads((Path(__file__).resolve().parents[1] / 'configs' / 'example.json').read_text())
    lidar = Lidar(profile['base_from_lidar_xyz_m'], profile['base_from_lidar_xyzw'], [0.0])
    rng = np.random.default_rng(seed)
    start0, dur = 500_000_000_000, 40.0
    rec = Recording(data / 'session')
    start = combined_pose(0)
    for k in range(int(dur * 50) + 1):
        e = k / 50
        rec.add(start0 + k * 20_000_000 + 2_000_000, '/demo/imu', IMU,
                imu_msg(start0 + k * 20_000_000, 'imu', rate(combined_pose, e), [0, 0, 0, 1], rng))
    for k in range(int(dur * 20) + 1):
        e = k / 20
        ns = start0 + k * 50_000_000
        moving = min(max(e - 1, 0), 19) + min(max(e - 24, 0), 4) + min(max(e - 30, 0), 9)
        local, yaw = drifted(combined_pose, e, start, moving)
        rec.add(ns + 2_000_000, '/demo/odom', ODOM, odom_msg(ns, 'odom', 'base_link', local, yaw))
    rows = {'mapping': [], 'inspection': [], 'all': []}
    for k in range(int((dur - 0.035) * 10) + 1):
        e = 0.035 + k / 10
        ns = start0 + int(round(e * 1e9))
        pose = combined_pose(e)
        r = ring_points(lidar, pose, rng)[0:240]
        keep = np.isfinite(r)
        xyz = lidar.dirs[keep] * r[keep, None]
        rec.add(ns + 2_000_000, '/demo/cloud', CLOUD, cloud_msg(ns, 'lidar', [
            ('x', 'f4', xyz[None, :, 0]), ('y', 'f4', xyz[None, :, 1]), ('z', 'f4', xyz[None, :, 2])]))
        rows['all'].append((ns / 1e9, pose))
        if e <= 20:
            rows['mapping'].append((ns / 1e9, pose))
        if e >= 30:
            rows['inspection'].append((ns / 1e9, pose))
    rec.close()
    for key, value in rows.items():
        write_truth(truth / f'{key}_truth.csv', value)
    (data / 'ORGANIZERS.md').write_text(
        'Одна запись `session/`: робот сначала проходит полный обход для карты, затем стоит, '
        'переезжает к точке старта инспекции и выполняет инспекцию. Требуется карта по обходу и '
        'положение робота на инспекции. Датчики и калибровка как в квалификационном задании: лидар '
        'в 0.2 м впереди и 0.3 м выше base_link без поворота, IMU выровнен с base_link.\n',
        encoding='utf-8')
    expected = dict(profile, mapping_header_s=[500.0, 520.0], inspection_header_s=[530.0, 540.0])
    (truth / 'expected_profile.json').write_text(json.dumps(expected, indent=2), encoding='utf-8')


# --------------------------------------------------------------------- scenario c
def scenario_c(root, seed):
    data, truth = root / 'data', root / 'truth'
    data.mkdir(parents=True)
    truth.mkdir()
    tilt = Rotation.from_euler('xyz', [0, 10, 0], degrees=True)  # nose down 10 deg
    lidar_xyz = [0.3, 0.0, 0.5]
    rings = np.linspace(-15, 15, 16)
    lidar = Lidar(lidar_xyz, tilt.as_quat(), rings, columns=360)
    sweep = 0.1
    col_time = lidar.column / lidar.shape[1] * sweep  # seconds after header stamp
    expected = dict(lidar='/velodyne_points', imu='/imu', odom='/odometry/filtered', lidar_frame='velodyne',
                    imu_frame='imu_link', base_frame='base_link', odom_frame='odom',
                    base_from_lidar_xyz_m=lidar_xyz, base_from_lidar_xyzw=tilt.as_quat().tolist(),
                    base_from_imu_xyzw=[0, 0, 0, 1], height_band_m=[0.15, 1.5], max_sync_gap_s=0.35,
                    stationary_s=0.5, point_time=dict(field='t', unit='ns'))
    for i, run in enumerate(('mapping', 'inspection')):
        rng = np.random.default_rng(seed + i)
        dur = 20.0 if run == 'mapping' else 10.0
        start0 = 1_000_000_000_000 * (i + 1)
        traj = lambda e, run=run: pose_at(run, e)  # noqa: E731
        start = traj(0.0)
        rec = Recording(data / run)
        for k in range(int(dur * 200) + 1):
            e = k / 200
            rec.add(start0 + k * 5_000_000 + 500_000, '/imu', IMU,
                    imu_msg(start0 + k * 5_000_000, 'imu_link', rate(traj, e), [0, 0, 0, 1], rng))
        for k in range(int(dur * 50) + 1):
            e = k / 50
            ns = start0 + k * 20_000_000
            local, yaw = drifted(traj, e, start, np.clip(e - 1, 0, dur - 2))
            rec.add(ns + 1_000_000, '/odometry/filtered', ODOM, odom_msg(ns, 'odom', 'base_link', local, yaw))
        rows = []
        for k in range(int((dur - sweep - 0.01) * 10) + 1):
            e = 0.01 + k / 10
            ns = start0 + int(round(e * 1e9))
            poses = np.array([traj(e + t) for t in col_time])
            r = lidar.ranges(poses)
            r = r + rng.normal(0, 0.01, r.shape)
            r[rng.random(r.shape) < 0.05] = np.inf
            ok = np.isfinite(r)
            xyz = lidar.dirs * np.where(ok, r, 0)[:, None]
            xyz[~ok] = np.nan
            shape = lidar.shape
            ring = np.repeat(np.arange(shape[0]), shape[1]).reshape(shape)
            t_ns = np.round(col_time * 1e9).astype(np.uint32).reshape(shape)
            rec.add(ns + int(sweep * 1e9) + 4_000_000, '/velodyne_points', CLOUD, cloud_msg(ns, 'velodyne', [
                ('x', 'f8', xyz[:, 0].reshape(shape)), ('y', 'f8', xyz[:, 1].reshape(shape)),
                ('z', 'f8', xyz[:, 2].reshape(shape)), ('intensity', 'f4', np.full(shape, 30.0)),
                ('t', 'u4', t_ns), ('ring', 'u2', ring)]))
            rows.append((ns / 1e9, traj(e)))
        rec.close()
        write_truth(truth / f'{run}_truth.csv', rows)
    (data / 'calibration.yaml').write_text(
        '# Калибровка, измеренная на стенде. Перенос из системы датчика в base_link.\n'
        'velodyne:\n  parent: base_link\n  xyz_m: [0.30, 0.00, 0.50]\n'
        '  rpy_deg: [0.0, 10.0, 0.0]   # roll, pitch, yaw; лидар наклонён вперёд-вниз\n'
        'imu_link:\n  parent: base_link\n  xyz_m: [0.00, 0.00, 0.20]\n  rpy_deg: [0.0, 0.0, 0.0]\n',
        encoding='utf-8')
    (data / 'ORGANIZERS.md').write_text(
        'Две записи: `mapping/` — обход, `inspection/` — инспекция. 16-лучевой лидар наклонён вниз, '
        'облака организованные (кольцо x азимут), пропуски отражений — NaN. Поле `t` — время точки '
        'в наносекундах от метки заголовка, заголовок ставится на начало оборота. Калибровка в '
        '`calibration.yaml`, /tf_static в записях нет. В начале каждой записи робот стоит около 1 с.\n',
        encoding='utf-8')
    (truth / 'expected_profile.json').write_text(json.dumps(expected, indent=2), encoding='utf-8')


# --------------------------------------------------------------------- scenario d
def scenario_d(root, seed):
    data, truth = root / 'data', root / 'truth'
    data.mkdir(parents=True)
    truth.mkdir()
    flip = Rotation.from_euler('x', 180, degrees=True)  # mounted upside down
    lidar_xyz = [0.1, 0.0, 0.35]
    lidar = Lidar(lidar_xyz, flip.as_quat(), [0.0])
    odom_lead_s = 0.25
    gaps = {'mapping': [(6.0, 6.6), (13.0, 14.2)], 'inspection': [(4.0, 4.6)]}
    expected = dict(lidar='/scan_cloud', imu=None, odom='/wheel/odometry', lidar_frame='laser',
                    base_frame='base_link', odom_frame='odom', base_from_lidar_xyz_m=lidar_xyz,
                    base_from_lidar_xyzw=flip.as_quat().tolist(), height_band_m=[0.15, 1.5],
                    max_sync_gap_s=0.35, stationary_s=0, header_offset_s=dict(odom=-odom_lead_s))
    for i, run in enumerate(('mapping', 'inspection')):
        rng = np.random.default_rng(seed + i)
        dur = 20.0 if run == 'mapping' else 10.0
        boot0 = 812_400_000_000 + i * 400_000_000_000   # robot boot clock in headers
        wall0 = 1_764_160_000_000_000_000 + i * 400_000_000_000
        traj = lambda e, run=run: pose_at(run, e)  # noqa: E731
        start = traj(0.0)
        rec = Recording(data / run)
        rec.add(wall0, '/tf_static', TF, T[TF]([transform(boot0, 'base_link', 'laser', lidar_xyz, flip.as_quat())]))
        for k in range(int(dur * 25) + 1):
            e = k / 25
            if any(a <= e < b for a, b in gaps[run]):
                continue
            local, yaw = drifted(traj, e, start, np.clip(e - 1, 0, dur - 2))
            v = np.linalg.norm(traj(e + 1e-3)[:2] - traj(e - 1e-3)[:2]) / 2e-3
            hdr = boot0 + k * 40_000_000 + int(odom_lead_s * 1e9)
            rec.add(wall0 + k * 40_000_000 + 5_000_000, '/wheel/odometry', ODOM,
                    odom_msg(hdr, 'odom', 'base_link', local, yaw, (v, rate(traj, e)), 0.01 + 0.002 * e))
        rows = []
        for k in range(int((dur - 0.02) * 10) + 1):
            e = 0.02 + k / 10
            pose = traj(e)
            r = ring_points(lidar, pose, rng)
            keep = np.isfinite(r)
            xyz = lidar.dirs[keep] * r[keep, None]
            hdr = boot0 + int(round(e * 1e9))
            rec.add(wall0 + int(round(e * 1e9)) + 30_000_000, '/scan_cloud', CLOUD, cloud_msg(hdr, 'laser', [
                ('x', 'f4', xyz[None, :, 0]), ('y', 'f4', xyz[None, :, 1]), ('z', 'f4', xyz[None, :, 2])]))
            rows.append((hdr / 1e9, pose))
        rec.close()
        write_truth(truth / f'{run}_truth.csv', rows)
    (data / 'ORGANIZERS.md').write_text(
        'Две записи: `mapping/` — обход, `inspection/` — инспекция. Робот без IMU, одометрия колёсная. '
        'Калибровка лидара в /tf_static.\n', encoding='utf-8')
    (truth / 'expected_profile.json').write_text(json.dumps(expected, indent=2), encoding='utf-8')


# --------------------------------------------------------------------- scenario e
def scenario_e(root, seed):
    data, truth = root / 'data', root / 'truth'
    data.mkdir(parents=True)
    truth.mkdir()
    tilt = Rotation.from_euler('xyz', [0, 6, 0], degrees=True)  # nose down 6 deg
    lidar_xyz = [0.2, 0.0, 0.35]
    lidar = Lidar(lidar_xyz, tilt.as_quat(), [0.0], columns=360)
    expected = dict(lidar='/scan_cloud', imu='/imu', odom='/odom', lidar_frame='laser', imu_frame='imu',
                    base_frame='base_link', odom_frame='odom', base_from_lidar_xyz_m=lidar_xyz,
                    base_from_lidar_xyzw=tilt.as_quat().tolist(), base_from_imu_xyzw=[0, 0, 0, 1],
                    height_band_m=[0.15, 1.5], max_sync_gap_s=0.35, stationary_s=0.5)
    for i, run in enumerate(('mapping', 'inspection')):
        rng = np.random.default_rng(seed + i)
        dur = 20.0 if run == 'mapping' else 10.0
        start0 = 300_000_000_000 * (i + 1)
        traj = lambda e, run=run: pose_at(run, e)  # noqa: E731
        start = traj(0.0)
        rec = Recording(data / run)
        for k in range(int(dur * 100) + 1):
            e = k / 100
            rec.add(start0 + k * 10_000_000 + 1_000_000, '/imu', IMU,
                    imu_msg(start0 + k * 10_000_000, 'imu', rate(traj, e), [0, 0, 0, 1], rng))
        for k in range(int(dur * 20) + 1):
            e = k / 20
            ns = start0 + k * 50_000_000
            local, yaw = drifted(traj, e, start, np.clip(e - 1, 0, dur - 2))
            rec.add(ns + 2_000_000, '/odom', ODOM, odom_msg(ns, 'odom', 'base_link', local, yaw))
        rows = []
        for k in range(int((dur - 0.03) * 10) + 1):
            e = 0.03 + k / 10
            ns = start0 + int(round(e * 1e9))
            pose = traj(e)
            r = ring_points(lidar, pose, rng, dropout=0.05)
            keep = np.isfinite(r)
            xyz = lidar.dirs[keep] * r[keep, None]
            rec.add(ns + 5_000_000, '/scan_cloud', CLOUD, cloud_msg(ns, 'laser', [
                ('x', 'f4', xyz[None, :, 0]), ('y', 'f4', xyz[None, :, 1]), ('z', 'f4', xyz[None, :, 2])]))
            rows.append((ns / 1e9, pose))
        rec.close()
        write_truth(truth / f'{run}_truth.csv', rows)
    (data / 'ORGANIZERS.md').write_text(
        'Две записи: `mapping/` — обход, `inspection/` — инспекция. Однолучевой лидар в 0.2 м впереди и '
        '0.35 м выше base_link, наклонён вперёд-вниз на 6°. IMU выровнен с base_link. /tf_static нет. '
        'В начале каждой записи робот стоит около 1 с.\n', encoding='utf-8')
    (truth / 'expected_profile.json').write_text(json.dumps(expected, indent=2), encoding='utf-8')


SCENARIOS = dict(a=scenario_a, b=scenario_b, c=scenario_c, d=scenario_d, e=scenario_e)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out', type=Path, help='New directory')
    parser.add_argument('--only', choices=SCENARIOS, nargs='*', default=list(SCENARIOS))
    parser.add_argument('--seed', type=int, default=23)
    args = parser.parse_args()
    for key in args.only:
        target = args.out / key
        if target.exists():
            parser.exit(2, f'{target} exists; choose a new directory\n')
        SCENARIOS[key](target, args.seed)
        print(target)


if __name__ == '__main__':
    main()
