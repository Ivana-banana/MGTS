"""Full-cloud deskew and bounded scan-to-map registration for the Go2 bags."""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

MAX_GAP_S = 0.10
MAX_MATCH_M = 0.35


@dataclass
class Pose:
    rotation: Rotation
    xyz: np.ndarray

    def apply(self, points):
        return self.rotation.apply(points) + self.xyz

    def compose(self, child):
        return Pose(self.rotation * child.rotation, self.apply(child.xyz))


def alignment_argument(parser):
    parser.add_argument("--map-from-odom", type=float, nargs=4,
                        metavar=("X", "Y", "Z", "YAW_RAD"),
                        help="Known inspection odom-to-map transform, metres and radians")


def alignment_pose(values):
    if values is None:
        return None
    if not np.isfinite(values).all():
        raise ValueError("--map-from-odom requires finite XYZ and yaw")
    return Pose(Rotation.from_euler("z", values[3]), np.array(values[:3]))


class CorrectionFilter:
    """Smooth the map correction while retaining odometry's short-term motion."""

    def __init__(self):
        self.previous_t = None
        self.shift = np.zeros(3)
        self.rotation = Rotation.identity()

    def update(self, t, odom, fit, accepted):
        dt = 0. if self.previous_t is None else t - self.previous_t
        self.previous_t = t
        if accepted:
            gain = -np.expm1(-dt / 0.5)
            self.shift += gain * (fit.xyz - odom.xyz - self.shift)
            target = fit.rotation * odom.rotation.inv()
            delta = (target * self.rotation.inv()).as_rotvec()
            self.rotation = Rotation.from_rotvec(gain * delta) * self.rotation
        return Pose(self.rotation * odom.rotation, odom.xyz + self.shift)


class Motion:
    def __init__(self, timestamps, xyz, quaternions):
        self.times = np.asarray(timestamps)
        self.xyz = np.asarray(xyz)
        self.origin = self.times[0]
        self.rotation = Slerp(self.times - self.origin, Rotation.from_quat(quaternions))

    def at(self, t):
        return Pose(self.rotation(t - self.origin),
                    np.stack([np.interp(t, self.times, axis) for axis in self.xyz.T], axis=-1))

    def covers(self, start, end):
        if start < self.times[0] or end > self.times[-1]:
            return False
        lo = np.searchsorted(self.times, start, side='right') - 1
        hi = np.searchsorted(self.times, end, side='left') + 1
        return bool(np.all(np.diff(self.times[lo:hi]) <= MAX_GAP_S))

    def deskew(self, timestamp, points, offsets):
        ref = self.at(timestamp)
        point_poses = self.at(timestamp + np.asarray(offsets, dtype=np.float64))
        return ref.rotation.inv().apply(point_poses.apply(points) - ref.xyz)


def voxel_centroids(points, size):
    _, index, counts = np.unique(np.floor(points / size).astype(np.int32),
                                axis=0, return_inverse=True, return_counts=True)
    return np.column_stack([np.bincount(index, weights=coord) for coord in points.T]) / counts[:, None]


class SurfaceMap:
    def __init__(self, points):
        self.tree = cKDTree(points)
        distances, indices = self.tree.query(points, k=12)
        cells = points[indices]
        centered = cells - cells.mean(axis=1, keepdims=True)
        values, vectors = np.linalg.eigh(centered.transpose(0, 2, 1) @ centered)
        self.normals = vectors[:, :, 0]
        self.planar = (values[:, 0] < 0.25 * values[:, 1]) & (distances[:, -1] <= 0.60)
        self.plane_tree = cKDTree(points[self.planar])
        self.plane_normals = self.normals[self.planar]

    def score(self, points, pose):
        if not len(points):
            return float('nan')
        distances, _ = self.tree.query(pose.apply(points))
        return float(np.mean(np.minimum(distances, MAX_MATCH_M)))

    def align(self, points, initial):
        fit = self._align(points, initial, self.tree, self.normals, self.planar)
        if fit is None and self.plane_tree.n:
            # An edge or isolated return can hide a usable plane at the same distance scale.
            fit = self._align(points, initial, self.plane_tree, self.plane_normals, None)
        return (initial, False) if fit is None else fit

    def _align(self, points, initial, tree, normals, valid):
        pose = initial
        for _ in range(6):
            rotated = pose.rotation.apply(points)
            world = rotated + pose.xyz
            distances, indices = tree.query(world)
            mask = distances < MAX_MATCH_M
            if valid is not None:
                mask &= valid[indices]
            if np.count_nonzero(mask) < max(35, 0.25 * len(points)):
                return None
            normal = normals[indices[mask]]
            error = np.einsum('ij,ij->i', world[mask] - tree.data[indices[mask]], normal)
            scale = max(1., np.median(np.linalg.norm(rotated[mask], axis=1)))
            jacobian = np.column_stack((normal, np.cross(rotated[mask], normal) / scale))
            weight = 1 / np.sqrt(1 + (error / 0.08) ** 2)
            system = jacobian * weight[:, None]
            gradient = system.T @ (weight * error)
            values, vectors = np.linalg.eigh(system.T @ system)
            observable = values >= 0.008 * values[-1]
            axes = vectors[:, observable]
            delta = -axes @ ((axes.T @ gradient) / values[observable])
            pose = Pose(Rotation.from_rotvec(delta[3:] / scale) * pose.rotation,
                        pose.xyz + delta[:3])
            if np.linalg.norm(delta) < 0.001:
                break
        shift = np.linalg.norm(pose.xyz - initial.xyz)
        turn = (pose.rotation * initial.rotation.inv()).magnitude()
        if shift > 0.60 or turn > 0.35 or self.score(points, pose) > self.score(points, initial) + 0.01:
            return initial, False
        return pose, True


def odometry_sample(message):
    pose = message.pose.pose
    p, q = pose.position, pose.orientation
    return [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]


def scan_points(points, offsets):
    distance = np.linalg.norm(points, axis=1)
    keep = np.isfinite(offsets) & (distance >= 0.25) & (distance <= 12.)
    return points[keep], offsets[keep]


def scans(path):
    from .bag import cloud_field, cloud_xyz, reader, stamp

    with reader(path) as bag:
        topics = {conn.topic: conn for conn in bag.connections}
        odom = topics['/utlidar/robot_odom']
        samples = []
        times = []
        for _, _, raw in bag.messages(connections=[odom]):
            message = bag.deserialize(raw, odom.msgtype)
            samples.append(odometry_sample(message))
            times.append(stamp(message))
        motion = Motion(times, *zip(*samples))
        lidar = topics['/utlidar/cloud_base']
        for _, _, raw in bag.messages(connections=[lidar]):
            message = bag.deserialize(raw, lidar.msgtype)
            t = stamp(message)
            points, offsets = scan_points(cloud_xyz(message), cloud_field(message, 'time', {7: 'f4'}))
            if len(points) and motion.covers(t + min(0., float(offsets.min())),
                                              t + max(0., float(offsets.max()))):
                started = perf_counter()
                yield t, motion.deskew(t, points, offsets), motion.at(t), started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mapping', type=Path)
    parser.add_argument('inspection', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    alignment_argument(parser)
    args = parser.parse_args()
    alignment = alignment_pose(args.map_from_odom)
    args.out.mkdir(parents=True)
    started = perf_counter()
    clouds = [pose.apply(points) for _, points, pose, _ in scans(args.mapping)]
    points = voxel_centroids(np.concatenate(clouds), 0.12)
    np.savez_compressed(args.out / 'map.npz', xyz=points)
    surface = SurfaceMap(points)
    map_s = perf_counter() - started
    rows = []
    filt = CorrectionFilter()
    odom_positions = []
    raw_positions = []
    for t, cloud, initial, started in scans(args.inspection):
        if alignment is not None:
            initial = alignment.compose(initial)
        held_out = np.arange(len(cloud)) % 4 == 0
        raw, accepted = surface.align(cloud[~held_out], initial)
        pose = filt.update(t, initial, raw, accepted)
        elapsed_ms = (perf_counter() - started) * 1000
        validation = cloud[held_out]
        odom_positions.append(initial.xyz)
        raw_positions.append(raw.xyz)
        rows.append([t, *pose.xyz, *pose.rotation.as_quat(), 'matched' if accepted else 'prediction_only',
                     elapsed_ms, surface.score(validation, initial), surface.score(validation, raw),
                     surface.score(validation, pose)])
    with (args.out / 'trajectory.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'status',
                         'compute_ms', 'odom_residual_m', 'raw_residual_m', 'fit_residual_m'])
        writer.writerows(rows)
    steps_p95 = {}
    for label, xyz in [('odom', odom_positions), ('raw', raw_positions),
                       ('filtered', [row[1:4] for row in rows])]:
        steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        steps_p95[label] = float(np.percentile(steps, 95))
    stats = {
        'mapping_scans': len(clouds),
        'map_voxels': len(points),
        'mapping_s': map_s,
        'inspection_scans': len(rows),
        'matched': sum(row[8] == 'matched' for row in rows),
        'compute_p95_ms': float(np.percentile([row[9] for row in rows], 95)),
        'heldout_mean_odom_residual_m': float(np.mean([row[10] for row in rows])),
        'heldout_mean_raw_residual_m': float(np.mean([row[11] for row in rows])),
        'heldout_mean_fit_residual_m': float(np.mean([row[12] for row in rows])),
        'position_step_p95_m': steps_p95,
    }
    (args.out / 'summary.json').write_text(json.dumps(stats, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
