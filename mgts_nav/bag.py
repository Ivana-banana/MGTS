"""ROS boundary. All motion and scan timestamps below use header time (seconds)."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader, AnyReaderError
from rosbags.serde.errors import SerdeError
from rosbags.typesys.base import TypesysError
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

from .deskew import deskew_points
from .leveling import build_leveler


def reader(path):
    return AnyReader([Path(path)], default_typestore=get_typestore(Stores.ROS2_HUMBLE))


def stamp(msg):
    s = msg.header.stamp
    return s.sec + s.nanosec * 1e-9


def inspect_bag(path, inventory=False):
    """Adapted from Transport_MSK/tools/survey/survey_bags.py; no tram message types."""
    rows = []
    with reader(path) as bag:
        for conn in bag.connections:
            times, lags, frames = [], [], set()
            info = dict(topic=conn.topic, type=conn.msgtype, messages=conn.msgcount)
            if inventory:
                rows.append(info)
                continue
            for _, t, raw in bag.messages(connections=[conn]):
                try:
                    msg = bag.deserialize(raw, conn.msgtype)
                except (KeyError, ValueError, TypeError, TypesysError, SerdeError) as exc:
                    info['decode_error'] = str(exc)
                    break
                if not hasattr(msg, 'header'):
                    break
                s = stamp(msg)
                times.append(s)
                lags.append(1000 * (t * 1e-9 - s))
                frames.add(msg.header.frame_id)
                if hasattr(msg, 'fields') and 'fields' not in info:
                    info['fields'] = [f.name for f in msg.fields]
            if times:
                dt = np.diff(times)
                positive = dt[dt > 0]
                info.update(frames=sorted(frames), header_start_s=times[0], header_end_s=times[-1],
                            median_hz=float(1 / np.median(positive)) if positive.size else None,
                            max_gap_s=float(max(positive)) if positive.size else None,
                            backwards=int(sum(dt < 0)), duplicates=int(sum(dt == 0)),
                            median_record_minus_header_ms=float(np.median(lags)))
            rows.append(info)
    return rows


def cloud_field(msg, name, formats):
    order = '>' if msg.is_bigendian else '<'
    fields = {f.name: f for f in msg.fields}
    f = fields.get(name)
    if f is None or f.datatype not in formats or f.count != 1:
        raise ValueError(f'PointCloud2 needs a scalar field {name!r} with supported type')
    dtype = np.dtype(order + formats[f.datatype])
    if f.offset < 0 or f.offset + dtype.itemsize > msg.point_step or msg.row_step < msg.width * msg.point_step:
        raise ValueError('Invalid PointCloud2 strides/offsets')
    if not msg.width or not msg.height:
        return np.empty(0)
    return np.ndarray((msg.height, msg.width), dtype, buffer=msg.data,
                      offset=f.offset, strides=(msg.row_step, msg.point_step)).ravel()


def cloud_xyz(msg):
    return np.column_stack([cloud_field(msg, name, {7: 'f4', 8: 'f8'}) for name in 'xyz'])


def point_offsets(msg, spec, finite):
    """Seconds from the raw lidar header; clock correction cancels in this difference."""
    mode = spec.get('semantics', 'signed')
    formats = {8: 'f8'} if mode == 'absolute' else {3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
    values = cloud_field(msg, spec['field'], formats)[finite].astype(float)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite per-point time offsets')
    if mode == 'absolute':
        return values - stamp(msg)
    scale = {'s': 1.0, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9}[spec['unit']]
    if mode == 'backward':
        if np.any(values < 0):
            raise ValueError('Backward point_time durations must be nonnegative')
        scale = -scale
    return values * scale


def connections(bag, cfg):
    result = {}
    for key, typ in [('lidar', 'sensor_msgs/msg/PointCloud2'), ('imu', 'sensor_msgs/msg/Imu'),
                     ('odom', 'nav_msgs/msg/Odometry')]:
        if key != 'lidar' and cfg[key] is None:
            result[key] = []
            continue
        hits = [c for c in bag.connections if c.topic == cfg[key]]
        if not hits or any(c.msgtype != typ for c in hits):
            raise ValueError(f'Expected {typ} on {cfg[key]}. Run inspect and check the profile.')
        result[key] = hits
    return result


def check_frame(actual, expected):
    if actual != expected:
        raise ValueError(f'Unexpected frame {actual!r}, expected {expected!r}; check calibration.')


def load_config(path):
    def unique_fields(pairs):
        fields = {}
        for key, value in pairs:
            if key in fields:
                raise ValueError(f'{path}: duplicate profile field {key!r}')
            fields[key] = value
        return fields

    cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique_fields)
    if not isinstance(cfg, dict) or cfg.get('status') == 'draft':
        raise ValueError(f'{path}: expected a sensor profile. Complete the draft and save its profile object.')
    if 'gravity_level' in cfg and not isinstance(cfg['gravity_level'], bool):
        raise ValueError(f'{path}: gravity_level must be JSON true or false')
    names = ['lidar', 'lidar_frame']
    vectors = {'height_band_m': 2, 'base_from_lidar_xyz_m': 3, 'base_from_lidar_xyzw': 4}
    for role in ('imu', 'odom'):
        if role not in cfg:
            raise ValueError(f'{path}: set {role} to a topic name or null')
        if cfg[role] is not None:
            names.extend([role, role + '_frame'])
    if cfg['imu'] is not None:
        vectors['base_from_imu_xyzw'] = 4
    if cfg['odom'] is not None:
        names.append('base_frame')
    for key in names:
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError(f'{path}: supply a topic or frame name for {key}')

    def finite_number(value):
        return type(value) in (int, float) and np.isfinite(value)

    for key, size in vectors.items():
        value = cfg.get(key)
        if not isinstance(value, list) or len(value) != size or not all(map(finite_number, value)):
            raise ValueError(f'{path}: {key} needs {size} finite numbers')
        if size == 4 and np.linalg.norm(value) == 0:
            raise ValueError(f'{path}: {key} has a zero quaternion')
    for key in ('max_sync_gap_s', 'stationary_s'):
        if not finite_number(cfg.get(key)):
            raise ValueError(f'{path}: {key} needs a finite duration in seconds')
    if cfg['max_sync_gap_s'] <= 0 or cfg['stationary_s'] < 0:
        raise ValueError('Require max_sync_gap_s > 0 and stationary_s >= 0')
    if not cfg['height_band_m'][0] < cfg['height_band_m'][1]:
        raise ValueError('Invalid height band')
    offsets = cfg.get('header_offset_s', {})
    if not isinstance(offsets, dict):
        raise ValueError('header_offset_s must be an object with offsets in seconds')
    for key, value in offsets.items():
        if key not in ('lidar', 'imu', 'odom') or not finite_number(value):
            raise ValueError('header_offset_s takes finite offsets for lidar/imu/odom')
    point_time = cfg.get('point_time')
    if point_time is not None:
        if (not isinstance(point_time, dict) or not isinstance(point_time.get('field'), str)
                or not point_time['field'].strip() or point_time.get('unit') not in ('s', 'ms', 'us', 'ns')):
            raise ValueError('point_time requires a field name and unit s/ms/us/ns')
        mode = point_time.get('semantics', 'signed')
        if set(point_time) - {'field', 'unit', 'semantics'} or mode not in ('signed', 'backward', 'absolute'):
            raise ValueError('point_time supports field, unit and semantics signed/backward/absolute')
        if mode == 'absolute' and point_time['unit'] != 's':
            raise ValueError('Absolute point_time requires float64 seconds on the raw lidar header clock')
    if point_time is not None and cfg['odom'] is None:
        raise ValueError('Deskew requires a kinematic odometry topic')
    if cfg.get('gravity_level') and cfg['imu'] is None:
        raise ValueError('Gravity leveling requires an IMU topic')
    return cfg


def check_recording(path, profile):
    """Validate a sensor profile against bag metadata before processing either route."""
    cfg = load_config(profile)
    try:
        with reader(path) as bag:
            for role, conns in connections(bag, cfg).items():
                if conns and not any(conn.msgcount for conn in conns):
                    raise ValueError(f'No messages on selected {role} topic {cfg[role]}')
    except (AnyReaderError, ValueError, OSError) as exc:
        raise ValueError(f'{path}: {exc}') from exc
    return cfg


@dataclass
class Scan:
    t: float
    xy: np.ndarray
    origin: np.ndarray
    odom: np.ndarray | None
    gyro_delta: float | None
    empty_reason: str | None = None


def interval_samples(times, start, end, gap):
    """Slice bracketing the interval. Disallow extrapolation and large gaps."""
    if not len(times) or start < times[0] or end > times[-1]:
        return None
    lo = times.searchsorted(start, side='right') - 1
    hi = times.searchsorted(end, side='left') + 1
    s = slice(lo, hi)
    if np.any(np.diff(times[s]) > gap):
        return None
    return s


def imu_values(msg, rotation, level_scans):
    has_rate = msg.angular_velocity_covariance[0] != -1
    w = msg.angular_velocity
    value = rotation.apply([w.x, w.y, w.z]).tolist() if has_rate else [0.] * 3
    has_tilt = False
    if level_scans:
        has_acc = msg.linear_acceleration_covariance[0] != -1
        acc = msg.linear_acceleration
        value.extend(rotation.apply([acc.x, acc.y, acc.z]) if has_acc else [0.] * 3)
        has_tilt = has_rate and has_acc
    return value, has_rate, has_tilt


def imu_rates(gyro, gyro_available, tilt_available, cfg):
    gt = gyro[:, 0]
    gap = cfg['max_sync_gap_s']
    gyro_available = np.asarray(gyro_available, dtype=bool)
    bias = np.zeros(3)
    if len(gt) and cfg['stationary_s'] > 0:
        end = gt[0] + cfg['stationary_s']
        if interval_samples(gt, gt[0], end, gap) is None:
            raise ValueError('Insufficient IMU coverage in configured stationary segment')
        samples = gyro[(gt <= end) & gyro_available, 1:4]
        bias = np.median(samples, axis=0) if len(samples) else np.full(3, np.nan)
    rates = gyro[:, 1:4] - bias
    rates[~gyro_available] = np.nan
    yaw_rate = rates[:, 2]
    leveler = None
    if cfg.get('gravity_level', False):
        leveler = build_leveler(gt, gyro[:, 1:4], gyro[:, 4:7], cfg['stationary_s'], gap,
                                available=tilt_available)
        # ZYX Euler yaw derivative for a body pitched and rolled relative to gravity.
        up = leveler.up_in_base[:, 1:]
        den = np.sum(up**2, axis=1)
        yaw_rate = np.divide(np.sum(up * rates[:, 1:], axis=1), den,
                             out=np.full_like(den, np.nan), where=den > 0.05)
    return yaw_rate, leveler


def scans(path, cfg):
    lidar_rot = Rotation.from_quat(cfg['base_from_lidar_xyzw'])
    offset = np.asarray(cfg['base_from_lidar_xyz_m'], dtype=float)
    imu_rot = Rotation.from_quat(cfg['base_from_imu_xyzw']) if cfg['imu'] is not None else None
    gap = cfg['max_sync_gap_s']
    clocks = cfg.get('header_offset_s', {})
    level_scans = cfg.get('gravity_level', False)
    with reader(path) as bag:
        conns = connections(bag, cfg)
        motion = {'odom': [], 'imu': []}
        gyro_available = []
        tilt_available = []
        for key in motion:
            if not conns[key]:
                motion[key] = np.empty((0, 4))
                continue
            for conn, _, raw in bag.messages(connections=conns[key]):
                m = bag.deserialize(raw, conn.msgtype)
                check_frame(m.header.frame_id, cfg[f'{key}_frame'])
                if key == 'imu':
                    value, has_rate, has_tilt = imu_values(m, imu_rot, level_scans)
                    gyro_available.append(has_rate)
                    if level_scans:
                        tilt_available.append(has_tilt)
                else:
                    check_frame(m.child_frame_id, cfg['base_frame'])
                    p, q = m.pose.pose.position, m.pose.pose.orientation
                    heading = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                    value = [p.x, p.y, np.arctan2(heading[1, 0], heading[0, 0])]
                motion[key].append([stamp(m) + clocks.get(key, 0), *value])
            a = np.asarray(motion[key])
            if not len(a) or not np.isfinite(a).all() or np.any(np.diff(a[:, 0]) <= 0):
                raise ValueError(f'Empty, non-finite or non-increasing {key}. Run inspect.')
            motion[key] = a
        odom, gyro = motion['odom'], motion['imu']
        odom[:, 3] = np.unwrap(odom[:, 3])
        gt = gyro[:, 0]
        yaw_rate, leveler = imu_rates(gyro, gyro_available, tilt_available, cfg)
        last = None
        for conn, _, raw in bag.messages(connections=conns['lidar']):
            m = bag.deserialize(raw, conn.msgtype)
            check_frame(m.header.frame_id, cfg['lidar_frame'])
            t = stamp(m) + clocks.get('lidar', 0)
            if last is not None and t <= last:
                raise ValueError('LiDAR header time went backwards or repeated')
            xyz = cloud_xyz(m)
            reason = None if len(xyz) else 'empty_input'
            finite = np.isfinite(xyz).all(axis=1)
            xyz = xyz[finite]
            if not len(xyz) and reason is None:
                reason = 'no_finite_points'
            xyz = xyz @ lidar_rot.as_matrix().T + offset
            point_time = cfg.get('point_time')
            if point_time is not None and len(xyz):
                elapsed = point_offsets(m, point_time, finite)
                begin, end = t + min(0, elapsed.min()), t + max(0, elapsed.max())
                span = interval_samples(odom[:, 0], begin, end, gap)
                if span is None:
                    reason = 'deskew_odom_unavailable'
                    xyz = xyz[:0]
                else:
                    xyz = deskew_points(xyz, elapsed, t, odom[span], gap)
            origin = offset
            if leveler is not None:
                try:
                    leveling = leveler.at(t)
                except ValueError:
                    reason = reason or 'leveling_unavailable'
                    xyz = xyz[:0]
                else:
                    xyz = xyz @ leveling.as_matrix().T
                    origin = leveling.apply(offset)
            low, high = cfg['height_band_m']
            xy = xyz[(xyz[:, 2] >= low) & (xyz[:, 2] <= high), :2]
            if len(xyz) and not len(xy):
                reason = 'outside_height_band'
            # One representative per voxel keeps dense sensors affordable.
            _, index = np.unique(np.floor(xy / 0.06), axis=0, return_index=True)
            prior = None
            if interval_samples(odom[:, 0], t, t, gap) is not None:
                prior = np.array([np.interp(t, odom[:, 0], col) for col in odom[:, 1:].T])
            delta = None
            if last is not None:
                s = interval_samples(gt, last, t, gap)
                if s is not None and np.isfinite(yaw_rate[s]).all():
                    inner = gt[s]
                    times = np.r_[last, inner[(inner > last) & (inner < t)], t]
                    delta = np.trapezoid(np.interp(times, gt[s], yaw_rate[s]), times)
            yield Scan(t, xy[index], origin[:2], prior, delta, reason)
            last = t
