"""Accuracy of the Go2 3D tracker on GrandTour HEAP-1 against the CPT7 reference.

ANYmal-D, Velodyne VLP-16, leg-inertial odometry, CPT7 GNSS/INS (tightly coupled).
The algorithm is mgts_nav.localize3d unchanged: voxel map 0.12 m, point-to-plane ICP,
correction filter, every 4th point held out. Two maps separate the algorithm from the
error it inherits from its inputs:

  clean  map scans placed with CPT7 poses (no inherited drift);
  odom   map scans placed with leg odometry, as on Go2 (the map inherits its drift).

Priors (leg odometry carried into the reference frame by yaw and offset at an anchor):

  clean          anchor at the inspection start: known start, as the shared map/odom
                 frame gives on Go2; drift accumulates only during the inspection.
  clean_far      anchor at the mapping start: drift since mapping enters the prior.
  odom           map and prior share the odometry frame anchored at the mapping start,
                 exactly the Go2 setup; error includes the drift baked into the map.

Scans are deskewed with leg odometry, as on Go2. `truth_pose_residual_m` places each
inspection scan at its CPT7 pose: a frame and calibration check of the clean map.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from mgts_nav.localize3d import CorrectionFilter, Motion, Pose, SurfaceMap, scan_points, voxel_centroids
from tools.grandtour_zarr import Archive

MAPPING = (0, 1500)
INSPECTION = (2389, 3060)


def extrinsic(transforms, parent, child):
    tf = transforms[child]
    if (tf['base_frame_id'], tf['child_frame_id']) != (parent, child):
        raise ValueError(f'Missing transform {parent} <- {child}')
    return np.array([tf['translation'][k] for k in 'xyz']), Rotation.from_quat([tf['rotation'][k] for k in 'xyzw'])


def reference(data, transforms):
    """Pose of `base` in ENU from the CPT7 IMU pose."""
    nav = Archive(data / 'cpt7_ie_tc_odometry.tar')
    if nav.attrs['frame_id'] != 'enu_origin':
        raise ValueError('Unexpected CPT7 frame')
    t, xyz, q = nav.array('timestamp'), nav.array('pose_pos'), Rotation.from_quat(nav.array('pose_orien'))
    box_t, box_q = extrinsic(transforms, 'base', 'box_base')
    imu_t, imu_q = extrinsic(transforms, 'box_base', 'cpt7_imu')
    base_imu_t, base_imu_q = box_t + box_q.apply(imu_t), box_q * imu_q
    enu_base = q * base_imu_q.inv()
    return Motion(t, xyz - enu_base.apply(base_imu_t), enu_base.as_quat())


def odometry(data):
    odom = Archive(data / 'anymal_state_odometry.tar')
    return Motion(odom.array('timestamp'), odom.array('pose_pos'), odom.array('pose_orien'))


def lidar_scans(data, transforms, motion, first, last):
    """(t, deskewed points in base at t) for scans first..last-1, like localize3d.scans."""
    lidar_t, lidar_q = extrinsic(transforms, 'base', 'velodyne_lidar')
    velodyne = Archive(data / 'velodyne_points.tar')
    stamps, valid = velodyne.array('timestamp'), velodyne.array('valid')[:, 0]
    rows = velodyne.meta('points')['chunks'][0]
    for chunk in range(first // rows, (last - 1) // rows + 1):
        points, times = velodyne.chunk('points', chunk), velodyne.chunk('time', chunk)
        for k in range(max(first, chunk * rows), min(last, (chunk + 1) * rows)):
            n, row = valid[k], k - chunk * rows
            xyz, offsets = scan_points(lidar_q.apply(points[row, :n]) + lidar_t, times[row, :n])
            t = stamps[k]
            if len(xyz) and motion.covers(t + min(0., offsets.min()), t + max(0., offsets.max())):
                yield t, motion.deskew(t, xyz, offsets)
        del points, times


def errors(estimate, truth):
    d = np.array([p.xyz - g.xyz for p, g in zip(estimate, truth)])
    yaw = np.array([(g.rotation.inv() * p.rotation).as_euler('ZYX')[0] for p, g in zip(estimate, truth)])
    xy = np.linalg.norm(d[:, :2], axis=1)
    return {'xy_rmse_m': float(np.sqrt(np.mean(xy ** 2))), 'xy_p95_m': float(np.percentile(xy, 95)),
            'xy_max_m': float(xy.max()), 'z_rmse_m': float(np.sqrt(np.mean(d[:, 2] ** 2))),
            'yaw_rmse_deg': float(np.degrees(np.sqrt(np.mean(yaw ** 2))))}


def run(data, out):
    transforms = yaml.safe_load((data / 'tf.yaml').read_text(encoding='utf-8'))
    truth, odom = reference(data, transforms), odometry(data)
    mapping = list(lidar_scans(data, transforms, odom, *MAPPING))
    # CPT7 ends about 2 s before the last scan: keep scans the reference covers.
    inspection = [s for s in lidar_scans(data, transforms, odom, *INSPECTION) if truth.covers(s[0], s[0])]
    # Reference frame origin at the first mapping scan; odometry carried in by yaw and offset only.
    t0 = mapping[0][0]
    g0, o0 = truth.at(t0), odom.at(t0)
    origin = g0.xyz.copy()

    def ref(t):
        g = truth.at(t)
        return Pose(g.rotation, g.xyz - origin)

    def anchored(anchor):
        g, o = ref(anchor), odom.at(anchor)
        turn = Rotation.from_euler('z', g.rotation.as_euler('ZYX')[0] - o.rotation.as_euler('ZYX')[0])
        return lambda t: Pose(turn * odom.at(t).rotation, turn.apply(odom.at(t).xyz - o.xyz) + g.xyz)

    from_mapping, from_inspection = anchored(t0), anchored(inspection[0][0])
    truth_inspection = [ref(t) for t, _ in inspection]
    near = [from_inspection(t) for t, _ in inspection]
    far = [from_mapping(t) for t, _ in inspection]
    report = {'mapping_scans': len(mapping), 'inspection_scans': len(inspection),
              'mapping_s': float(mapping[-1][0] - t0), 'inspection_s': float(inspection[-1][0] - inspection[0][0]),
              'odometry_drift_over_mapping_m': (from_mapping(mapping[-1][0]).xyz - ref(mapping[-1][0]).xyz).round(3).tolist(),
              'prior_from_inspection_start': errors(near, truth_inspection),
              'prior_from_mapping_start': errors(far, truth_inspection)}
    out.mkdir(parents=True)
    maps = {}
    for name, place, prior in (('clean', ref, near), ('clean_far', ref, far), ('odom', from_mapping, far)):
        key = place.__name__ if place is ref else 'odom'
        if key not in maps:
            maps[key] = voxel_centroids(np.concatenate([place(t).apply(c) for t, c in mapping]), 0.12)
        points = maps[key]
        surface, filt, poses, matched, residual = SurfaceMap(points), CorrectionFilter(), [], 0, []
        if name == 'clean':
            report['truth_pose_residual_m'] = float(np.mean(
                [surface.score(c[::4], g) for (_, c), g in zip(inspection, truth_inspection)]))
        rows = []
        for (t, cloud), initial, g in zip(inspection, prior, truth_inspection):
            held = np.arange(len(cloud)) % 4 == 0
            fit, accepted = surface.align(cloud[~held], initial)
            pose = filt.update(t, initial, fit, accepted)
            poses.append(pose)
            matched += accepted
            residual.append(surface.score(cloud[held], pose))
            rows.append([t, *pose.xyz, *g.xyz, int(accepted)])
        np.savetxt(out / f'{name}.csv', rows, delimiter=',', comments='',
                   header='t,x,y,z,ref_x,ref_y,ref_z,accepted', fmt='%.6f')
        report[name] = {'map_voxels': len(points), 'matched': int(matched),
                        'heldout_mean_residual_m': float(np.mean(residual)), **errors(poses, truth_inspection)}
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', type=Path, help='folder with the GrandTour HEAP-1 TAR archives and tf.yaml')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.out)
