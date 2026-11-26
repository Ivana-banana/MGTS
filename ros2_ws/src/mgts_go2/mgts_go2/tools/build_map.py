"""Build the 3D map from the mapping bag.

    ros2 run mgts_go2 build_map /ros2_ws/bag_for_mapping --out /ros2_ws/mgts_map
    python3 -m mgts_go2.tools.build_map BAG --out DIR          (without ROS, needs ``rosbags``)

Outputs in DIR: map.pcd (refined, dynamic objects removed), map_odom_raw.pcd
(same scans placed by raw odometry, for comparison), map2d.pgm/.yaml (height
slice for nav2), mapping_trajectory.csv, mapping_report.json.
"""
import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

from ..core import bagio, mapio, mapping


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cloud-topic', default=bagio.CLOUD_TOPIC)
    ap.add_argument('--odom-topic', default=bagio.ODOM_TOPIC)
    ap.add_argument('--every', type=int, default=1, help='use every N-th sweep')
    ap.add_argument('--passes', type=int, default=2)
    ap.add_argument('--map-voxel', type=float, default=0.05)
    ap.add_argument('--no-dynamic-filter', action='store_true')
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = mapping.MapParams(passes=args.passes, map_voxel=args.map_voxel)
    print('reading', args.bag)
    odom, scans, _ = bagio.load(args.bag, args.cloud_topic, args.odom_topic, every=args.every)
    print('  %d sweeps, %d odometry poses' % (len(scans), len(odom)))
    result, prepared = mapping.build(scans, odom, params)
    if args.no_dynamic_filter:
        result.points, result.intensity = mapping.assemble(prepared, result.poses, params)

    raw_pts, raw_int = mapping.assemble(prepared, result.odom_poses, params)
    mapio.write_pcd(out / 'map.pcd', result.points, result.intensity)
    mapio.write_pcd(out / 'map_odom_raw.pcd', raw_pts, raw_int)
    floor_z = float(np.median([p[2, 3] for p in result.poses]) - 0.31)
    mapio.write_grid(out / 'map2d', result.points, floor_z=floor_z)
    mapio.write_poses(out / 'mapping_trajectory.csv', result.stamps, result.poses)

    print('consistency checks (held-out sweeps vs map from the other sweeps)')
    cons_ref = mapping.consistency(prepared, result.poses, params)
    cons_raw = mapping.consistency(prepared, result.odom_poses, params)
    corr = [np.linalg.norm((np.linalg.inv(o) @ p)[:3, 3]) for p, o in zip(result.poses, result.odom_poses)]
    report = {
        'bag': str(args.bag),
        'params': dataclasses.asdict(params),
        'stats': result.stats,
        'consistency_refined': cons_ref,
        'consistency_odometry': cons_raw,
        'map_minus_odom_translation_m': {'median': float(np.median(corr)), 'max': float(np.max(corr)),
                                         'final': float(corr[-1])},
        'floor_z_estimate': floor_z,
        'bbox_min': result.points.min(axis=0).round(3).tolist(),
        'bbox_max': result.points.max(axis=0).round(3).tolist(),
    }
    mapio.write_json(out / 'mapping_report.json', report)
    for k in ('stats', 'consistency_refined', 'consistency_odometry', 'map_minus_odom_translation_m'):
        print(k, report[k])
    return 0


if __name__ == '__main__':
    sys.exit(main())
