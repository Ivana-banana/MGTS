"""Replay the #161 known-start lidar trajectory and score frozen #164 warnings."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.spatial import cKDTree

from mgts_nav.grid import read_map
from mgts_nav.metrics import read_csv
from mgts_nav.slam import track
from tools.drift_warning_trial import evaluate, predict
from tools.kitti_repeat import PREFIX, calibration, integrated, load_scan, timestamps


def run(source, run_dir):
    first = json.loads((run_dir/'scores.json').read_text())['inspection_first']
    ids = range(first, first + 40)
    with ZipFile(source/'drive.zip') as archive, ZipFile(source/'calib.zip') as calib:
        transform = calibration(calib)
        times = timestamps(archive, 'velodyne_points')
        imu_times = timestamps(archive, 'oxts')[ids]
        stream = [load_scan(archive, i, times[i], *transform) for i in ids]
        packets = np.array([np.fromstring(archive.read(PREFIX + f'oxts/data/{i:010d}.txt').decode(),
                                          sep=' ') for i in ids])
    # Integrate only body-frame velocities and rates from zero for the warning.
    odom = integrated(imu_times, packets, np.zeros(3))
    truth = read_csv(run_dir/'inspection-reference.csv')
    points = read_map(run_dir/'map.yaml')[2]
    tree = cKDTree(points)
    estimates = list(track(stream, np.array(truth[0][1:]), map_points=points))
    saved = read_csv(run_dir/'known.csv')
    if not all(row.t == pose.t and np.allclose(row.pose, pose[1:], rtol=0, atol=1e-8)
               for row, pose in zip(estimates, saved, strict=True)):
        raise ValueError('Replay differs from #161')
    results = {}
    for mode in ('absent', 'recorded', 'drift'):
        previous = previous_pose = None
        rows = []
        for i, (scan, row) in enumerate(zip(stream, estimates, strict=True)):
            gyro = odom[i, 2] - odom[i-1, 2] if i else None
            scan = replace(scan, odom=odom[i], gyro_delta=gyro)
            alerts = None
            if row.status == 'matched':
                alerts = predict(scan, previous, row.pose, previous_pose, row.inliers, tree, mode)
            rows.append((scan.t, row.pose, alerts))
            previous, previous_pose = scan, row.pose
        results[mode] = [evaluate(rows, truth, index) for index in (0, 1)]
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', type=Path)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.archives, args.run_dir), indent=2))
