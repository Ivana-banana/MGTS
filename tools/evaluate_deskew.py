"""Compare scan motion compensation through the normal ROS 2 bag and solve paths."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

from mgts_nav.bag import load_config, reader, scans
from mgts_nav.demo import ANGLES, DURATION, PROFILE, RAYS, START_NS, WALLS, cross, generate, pose_at
from mgts_nav.grid import read_map
from mgts_nav.map_eval import evaluate_map, segment_distances
from mgts_nav.metrics import evaluate, read_csv
from mgts_nav.slam import world
from mgts_nav.solve import solve


OFFSETS_US = np.linspace(-30_000, 60_000, len(ANGLES)).round().astype(np.int32)
MOUNT_YAW = np.deg2rad(20.)
POINT_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('offset', '<i4')])


def render_sweep(run, elapsed, profile, rng):
    """Intersect each emitted ray with the fixed office walls at its own acquisition pose."""
    x, y, yaw = pose_at(run, elapsed + OFFSETS_US * 1e-6)
    dx, dy, dz = profile['base_from_lidar_xyz_m']
    origins = np.column_stack((x + dx*np.cos(yaw) - dy*np.sin(yaw),
                               y + dx*np.sin(yaw) + dy*np.cos(yaw)))
    angles = ANGLES + MOUNT_YAW + yaw
    rays = np.column_stack((np.cos(angles), np.sin(angles)))
    edge = WALLS[:, 1] - WALLS[:, 0]
    delta = WALLS[None, :, 0] - origins[:, None]
    det = cross(rays[:, None], edge)
    with np.errstate(divide='ignore', invalid='ignore'):
        along_ray = cross(delta, edge) / det
        along_edge = cross(delta, rays[:, None]) / det
    valid = (np.abs(det) > 1e-10) & (along_ray > 0) & (along_edge >= 0) & (along_edge <= 1)
    nearest = np.where(valid, along_ray, np.inf).min(axis=1)
    keep = np.isfinite(nearest) & (rng.random(len(rays)) >= .1)
    distance = nearest[keep] + rng.normal(0., .012, keep.sum())
    points = np.empty(int(keep.sum()), dtype=POINT_DTYPE)
    points['x'], points['y'] = (distance[:, None] * RAYS[keep]).T
    points['z'] = .65 - dz
    points['offset'] = OFFSETS_US[keep]
    return points


def write_bag(source, target, run, seed, profile, *, unit='ns', sign=1, odom_gap=False):
    """Keep the original IMU, odometry and scan clock. Replace the instantaneous clouds."""
    scale = {'ns': 1000, 'us': 1}[unit] * sign
    rng = np.random.default_rng(seed)
    store = get_typestore(Stores.ROS2_HUMBLE)
    field = store.types['sensor_msgs/msg/PointField']
    fields = [field(name, offset, datatype, 1) for name, offset, datatype in
              [('x', 0, 7), ('y', 4, 7), ('z', 8, 7), ('offset', 12, 5)]]
    with reader(source) as bag, Writer(target, version=8) as writer:
        conns = {c.id: writer.add_connection(c.topic, c.msgtype, typestore=store) for c in bag.connections}
        for conn, received, raw in bag.messages():
            if conn.topic == profile['lidar'] or (odom_gap and conn.topic == profile['odom']):
                msg = bag.deserialize(raw, conn.msgtype)
                stamp = msg.header.stamp
                elapsed_ns = stamp.sec * 10**9 + stamp.nanosec - START_NS[run]
                if conn.topic == profile['lidar']:
                    points = render_sweep(run, elapsed_ns / 1e9, profile, rng)
                    points['offset'] *= scale
                    msg.fields, msg.width = fields, len(points)
                    msg.point_step, msg.row_step = POINT_DTYPE.itemsize, points.nbytes
                    msg.data = points.view(np.uint8)
                    raw = store.serialize_cdr(msg, conn.msgtype)
                    received += 60_000_000  # The last return arrives after header.stamp.
                elif 4_050_000_000 < elapsed_ns < 4_550_000_000:
                    continue
            writer.write(conns[conn.id], received, raw)


def sensor_profile():
    profile = load_config(PROFILE)
    profile['base_from_lidar_xyzw'] = Rotation.from_euler('z', MOUNT_YAW).as_quat().tolist()
    return profile


def scan_wall_rmse(bag, cfg, run, truth):
    """Score only; use reference-time truth to isolate point geometry from pose error."""
    groups = {'moving': [], 'stationary': []}
    for scan, ref in zip(scans(bag, cfg), truth, strict=True):
        if abs(scan.t - ref.t) > 1e-8:
            raise ValueError('Scan timestamp differs from reference')
        elapsed = scan.t - START_NS[run] / 1e9
        if elapsed < .93 or elapsed > DURATION[run] - .93:
            group = 'stationary'
        elif 1.07 < elapsed < DURATION[run] - 1.07:
            group = 'moving'
        else:
            continue  # Transition sweeps can include both regimes.
        xy = world(scan.xy, np.array([ref.x, ref.y, ref.yaw]))
        groups[group].append(segment_distances(xy, WALLS))
    return {name: float(np.sqrt(np.mean(np.concatenate(values)**2))) for name, values in groups.items()}


def compare_on_fixed_maps(directory, modes, truth):
    """Cross the two exported maps and inspection profiles, keeping the start fixed."""
    initial = modes['off']['initial']
    if modes['on']['initial'] != initial:
        raise ValueError('Different starting poses in paired runs')
    result = {}
    for map_mode, scan_mode in [('off', 'on'), ('on', 'off')]:
        target = directory / f'map-{map_mode}-scan-{scan_mode}'
        subprocess.run([sys.executable, '-m', 'mgts_nav', 'localize', directory / 'inspection',
                        '--config', directory / f'{scan_mode}.json', '--map', directory / map_mode / 'map/map.yaml',
                        '--initial', *map(str, initial), '--out', target], check=True)
        metrics = evaluate(read_csv(target / 'trajectory.csv'), truth)
        result[target.name] = dict(ate_m=metrics['ate_xy_rmse_m'], coverage=metrics['coverage'])
    return result


def compare(directory, seed):
    source = directory / 'instantaneous'
    generate(source, seed=seed)
    cfg = sensor_profile()
    for i, run in enumerate(('mapping', 'inspection')):
        write_bag(source / run, directory / run, run, seed + i, cfg)
    modes = {}
    initial = None
    for mode in ('off', 'on'):
        profile = cfg.copy()
        if mode == 'on':
            profile['point_time'] = dict(field='offset', unit='ns')
        config_path = directory / f'{mode}.json'
        config_path.write_text(json.dumps(profile), encoding='utf-8')
        result = directory / mode
        solve(directory / 'mapping', directory / 'inspection', config_path, result, initial=initial)
        summary = json.loads((result / 'inspection/summary.json').read_text())
        initial = summary['initial_pose']
        _, meta, pixels_xy = read_map(result / 'map/map.yaml')
        map_score = evaluate_map(pixels_xy, WALLS, meta['resolution'], tolerance=.12)
        modes[mode] = dict(initial=initial, map_occupied_wall_rmse_m=float(
            np.sqrt(np.mean(segment_distances(pixels_xy, WALLS)**2))),
            wall_length_coverage=map_score['static_wall_length_covered_fraction'])
        for run, folder in [('mapping', 'map'), ('inspection', 'inspection')]:
            truth = read_csv(source / f'{run}_truth.csv')
            metrics = evaluate(read_csv(result / folder / 'trajectory.csv'), truth)
            status = json.loads((result / folder / 'summary.json').read_text())['status']
            modes[mode][run] = dict(ate_m=metrics['ate_xy_rmse_m'], coverage=metrics['coverage'],
                                    status=status, scan_wall_rmse_m=scan_wall_rmse(directory / run, profile, run, truth))
    modes['crossed'] = compare_on_fixed_maps(directory, modes, read_csv(source / 'inspection_truth.csv'))
    return modes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path, help='New directory for bags, solve outputs and measured errors')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {seed: compare(args.out / str(seed), seed) for seed in (41, 73)}
    (args.out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
