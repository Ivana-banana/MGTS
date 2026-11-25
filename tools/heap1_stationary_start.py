"""Compare stock localize and coarse single-scan initialization on the fixed HEAP-1 holdout."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from mgts_nav.bag import load_config, reader, scans
from mgts_nav.coarse_start import estimate
from mgts_nav.grid import read_map
from mgts_nav.metrics import evaluate, read_csv, wrap


def cut(source, target, lidar, offset):
    # Keep the original IMU headers and the bracketing samples at both window edges.
    # Only the 60 requested lidar frames reach the navigation pipeline.
    store = get_typestore(Stores.ROS2_HUMBLE)
    index = 0
    with reader(source) as bag, Writer(target, version=8) as writer:
        outputs = {c.id: writer.add_connection(c.topic, c.msgtype, typestore=store) for c in bag.connections}
        for conn, t, raw in bag.messages():
            if conn.topic == lidar:
                index += 1
                if not offset < index <= offset + 60:
                    continue
            writer.write(outputs[conn.id], t, raw)
    if index != 180:
        raise ValueError('Expected exactly 180 inspection scans from heap1_revisit.py --held-out')


def compare(data, map_path, stage):
    profile = data / 'profile.json'
    config = load_config(profile)
    pixels, meta, _ = read_map(map_path)
    results = []
    for offset in range(0, 180, 60):
        bag = stage / f'window-{2620 + offset}'
        cut(data / 'inspection', bag, config['lidar'], offset)
        first = next(scans(bag, config))
        if not offset and abs(first.t - 1731588674.081123) > 1e-6:
            raise ValueError('Unexpected HEAP-1 holdout timestamps')
        for method in ('stock', 'coarse'):
            row = dict(first_frame=2620+offset, method=method)
            results.append(row)
            out = stage / f'{method}-{2620+offset}'
            cmd = [sys.executable, '-m', 'mgts_nav', 'localize', str(bag), '--map', str(map_path),
                   '--config', str(profile), '--out', str(out)]
            if method == 'coarse':
                try:
                    start = estimate(first, pixels, meta, {})
                except ValueError as exc:
                    row.update(outcome='refused', reason=str(exc))
                    continue
                cmd.extend(['--initial', *map(str, start)])
            run = subprocess.run(cmd, capture_output=True, text=True)
            if run.returncode == 2 and 'Provide --initial or check calibration.' in run.stderr:
                row.update(outcome='refused', reason=run.stderr.strip())
                continue
            if run.returncode:
                raise ValueError(f'Unexpected localize failure: {run.stderr}')
            row['result_dir'] = out.name

    reference = read_csv(data / 'inspection-reference.csv')
    if len(reference) != 180:
        raise ValueError('Expected 180 reference timestamps')
    for row in results:
        if 'result_dir' not in row:
            continue
        out = stage / row['result_dir']
        summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
        poses = read_csv(out / 'trajectory.csv')
        offset = row['first_frame']-2620
        truth = reference[offset:offset+60]
        if len(poses) != 60 or not np.allclose([p.t for p in poses], [r.t for r in truth], atol=1e-6, rtol=0):
            raise ValueError('Pose and reference clocks differ')
        start = np.array(summary['initial_pose'])
        xy = float(np.linalg.norm(start[:2]-truth[0][1:3]))
        angle = float(np.degrees(abs(wrap(start[2]-truth[0].yaw))))
        row.update(outcome='correct' if xy <= .25 and angle <= 10 else 'false_accept',
                   start_xy_m=xy, start_yaw_deg=angle, matched=summary['status'].get('matched', 0),
                   ate_m=evaluate(poses, truth)['ate_xy_rmse_m'])
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'map', 'out'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Choose a new output directory')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp)/'trial'
        stage.mkdir()
        report = json.dumps(compare(args.data, args.map, stage), indent=2, allow_nan=False)
        (stage/'report.json').write_text(report, encoding='utf-8')
        stage.rename(args.out)
        print(report)
