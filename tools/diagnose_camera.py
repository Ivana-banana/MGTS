"""Separate camera parallax, pixel noise and angular-source switching on the weak-lidar window."""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from rosbags.typesys import Stores, get_typestore

from mgts_nav.bag import load_config, scans
from mgts_nav.camera_demo import write_camera_pairs
from mgts_nav.demo import PROFILE, START_NS, WALLS, pose_at, record
from mgts_nav.slam import predict, relative
from mgts_nav.visual_fusion import apply_visual


def trajectory(motion):
    """Use the same office trajectory; hold the selected coordinates at their window midpoint."""
    def sample(run, elapsed, layout='office', duration=None):
        p = pose_at(run, elapsed, layout, duration)
        center = pose_at(run, 5., layout, duration)
        if motion == 'rotation':
            p[:2] = center[:2]
        elif motion == 'translation':
            p[2] = center[2]
        return p
    return sample


def metrics(measured, reference):
    usable = [(v, r) for v, r in zip(measured, reference, strict=True) if v is not None]
    if not usable:
        return 0, None, None
    diff = np.array([v-r for v, r in usable])
    deg = np.rad2deg(np.arctan2(np.sin(diff), np.cos(diff)))
    return len(deg), float(np.mean(deg)), float(np.sqrt(np.mean(deg**2)))


def deltas(stream):
    return [predict(np.zeros(3), a.odom, b)[2] for a, b in zip(stream, stream[1:])]


def weak_stream(directory, seed, sample, gap, store, cfg):
    directory.mkdir()
    record(directory, 'inspection', cfg, store, seed+1, camera_stress=gap, trajectory=sample)
    epoch = START_NS['inspection'] / 1e9
    return [s for s in scans(directory/'inspection', load_config(PROFILE)) if 4 <= s.t-epoch < 6]


def experiment(output, seeds):
    output.mkdir(parents=True, exist_ok=False)
    epoch_ns = START_NS['inspection']
    epoch = epoch_ns / 1e9
    store = get_typestore(Stores.ROS2_HUMBLE)
    cfg = json.loads(PROFILE.read_text(encoding='utf-8'))
    rows = []
    for seed in seeds:
        for motion in ('mixed', 'rotation', 'translation'):
            root = output / f'{seed}-{motion}'
            root.mkdir()
            sample = trajectory(motion)
            on = weak_stream(root/'gyro', seed, sample, False, store, cfg)
            gap = weak_stream(root/'gap', seed, sample, True, store, cfg)
            off = [replace(s, gyro_delta=None) for s in on]
            reference = np.diff([sample('inspection', s.t-epoch)[2] for s in on])
            for sigma in (0., .15):
                path = root / f'pairs-{sigma}.jsonl'
                write_camera_pairs(path, seed+2, sample, WALLS, epoch_ns, noise_px=sigma)
                # Retain exactly the four visual pairs for each of the 19 scored intervals.
                selected = root / f'window-{sigma}.jsonl'
                with path.open(encoding='utf-8', newline='') as src, \
                        selected.open('x', encoding='utf-8', newline='') as dst:
                    for line in src:
                        pair = json.loads(line)
                        if on[0].t <= pair['t0'] and pair['t1'] <= on[-1].t:
                            dst.write(line)
                camera = [s.gyro_delta for s in list(apply_visual(off, selected, {}))[1:]]
                for gyro, stream in [('on', on), ('off', off), ('gap', gap)]:
                    stats = {}
                    fused = list(apply_visual(stream, selected, stats))
                    result = dict(seed=seed, motion=motion, noise_px=sigma, gyro=gyro,
                                  intervals=len(reference), accepted_pairs=stats['accepted_pairs'],
                                  total_pairs=stats['accepted_pairs'] + stats['geometry_rejected_pairs'],
                                  blend=stats['blend'], fallback=stats['fallback'])
                    sources = dict(camera=camera, gyro=[s.gyro_delta for s in stream[1:]],
                                   odom=[relative(a.odom, b.odom)[2] for a,b in zip(stream, stream[1:])],
                                   baseline=deltas(stream), fused=deltas(fused))
                    for key, values in sources.items():
                        n, bias, rmse = metrics(values, reference)
                        result[key+'_n'], result[key+'_bias_deg'], result[key+'_rmse_deg'] = n, bias, rmse
                    rows.append(result)
            print(f'Completed {seed} {motion}', flush=True)
    with (output/'results.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[41, 73])
    args = parser.parse_args()
    experiment(args.out, args.seeds)
