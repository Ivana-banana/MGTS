"""Reproduce the two-seed camera ablation with the regular solve CLI and offline scoring."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from mgts_nav.demo import PROFILE, START_NS, generate
from mgts_nav.metrics import evaluate, read_csv


def compare(directory, seed):
    data = directory / 'data'
    generate(data, seed=seed, camera_experiment=True)
    modes = {}
    for mode in ('off', 'on'):
        args = [sys.executable, '-m', 'mgts_nav.solve', data / 'mapping', data / 'inspection',
                '--config', PROFILE, '--out', directory / mode]
        if mode == 'on':
            args.extend(['--inspection-visual-pairs', data / 'inspection-pairs.jsonl'])
        subprocess.run(args, check=True)

    truth = read_csv(data / 'inspection_truth.csv')
    start = START_NS['inspection'] / 1e9
    weak = [p for p in truth if 4 <= p.t - start < 6]
    for mode in ('off', 'on'):
        output = directory / mode / 'inspection'
        poses = read_csv(output / 'trajectory.csv')
        summary = json.loads((output / 'summary.json').read_text())
        modes[mode] = dict(route=evaluate(poses, truth), weak_lidar=evaluate(poses, weak),
                           initial_pose=summary['initial_pose'], status=summary['status'],
                           visual=summary.get('visual_yaw'))
    if modes['on']['initial_pose'] != modes['off']['initial_pose']:
        raise ValueError('The two runs started from different estimated poses')
    if (directory / 'on/map/map.pgm').read_bytes() != (directory / 'off/map/map.pgm').read_bytes():
        raise ValueError('Mapping results differ across runs')
    return modes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path, help='New directory for bags, solve outputs and metrics')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {seed: compare(args.out / str(seed), seed) for seed in (41, 73)}
    (args.out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
