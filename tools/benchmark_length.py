"""Measure mapping and localization on equal-prefix ROS 2 bags in Windows."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil
from rosbags.rosbag2 import Reader, Writer
from rosbags.typesys import Stores, get_typestore

from mgts_nav.benchmark import hardware
from mgts_nav.demo import DURATION, PROFILE, START_NS, TOPIC_TYPES, pose_at, record


def write_case(source, target, run, seconds, stride, store):
    stop = START_NS[run] + int(seconds * 1e9) + 2_000_001
    scans = points = 0
    with Reader(source / run) as reader, Writer(target / run, version=8) as writer:
        connections = {c.id: writer.add_connection(c.topic, c.msgtype, typestore=store)
                       for c in reader.connections}
        for c, t, raw in reader.messages(stop=stop):
            if c.msgtype == TOPIC_TYPES['lidar']:
                msg = store.deserialize_cdr(raw, c.msgtype)
                if stride > 1:
                    xyz = msg.data.reshape(msg.width, msg.point_step)[::stride].copy()
                    msg.width = len(xyz)
                    msg.row_step = msg.width * msg.point_step
                    msg.data = xyz.ravel()
                    raw = store.serialize_cdr(msg, c.msgtype)
                scans += 1
                points += msg.width
            writer.write(connections[c.id], t, raw)
    return dict(scans=scans, mean_points=points / scans,
                bag_bytes=sum(p.stat().st_size for p in (target / run).iterdir() if p.is_file()))


def measure(arguments, log):
    command = [sys.executable, '-m', 'mgts_nav', *map(str, arguments)]
    peak = 0
    started = time.perf_counter()
    with log.open('wb') as stream, subprocess.Popen(command, stdout=stream, stderr=stream) as worker:
        root = psutil.Process(worker.pid)
        while worker.poll() is None:
            try:
                processes = [root, *root.children(recursive=True)]
            except psutil.NoSuchProcess:
                break
            rss = 0
            for process in processes:
                try:
                    rss += process.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, rss)
            time.sleep(.02)
        if worker.wait():
            raise RuntimeError(f'Navigation failed; see {log}')
    return dict(wall_s=time.perf_counter() - started, peak_tree_rss_mib=peak / 2**20)


def prefix_error(short, long):
    def read(directory):
        with (directory / 'trajectory.csv').open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    a, b = read(short), read(long)
    if not a or len(b) < len(a):
        raise ValueError('Missing common trajectory prefix')
    deltas = []
    statuses = 0
    for first, second in zip(a, b):
        if float(first['t']) != float(second['t']):
            raise ValueError('Prefix timestamps differ')
        delta = np.array([float(first[k]) - float(second[k]) for k in ('x', 'y', 'yaw')])
        deltas.append((np.linalg.norm(delta[:2]), abs(np.arctan2(np.sin(delta[2]), np.cos(delta[2])))))
        statuses += first['status'] != second['status']
    xy, yaw = np.max(deltas, axis=0)
    return dict(samples=len(a), max_xy_m=xy, max_yaw_rad=yaw, status_mismatches=statuses)


def benchmark(output):
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    source.mkdir()
    store = get_typestore(Stores.ROS2_HUMBLE)
    profile = json.loads(PROFILE.read_text(encoding='utf-8'))
    for i, run in enumerate(DURATION):
        record(source, run, profile, store, seed=17+i, duration=DURATION[run]*5)
    results, comparisons = [], []
    for stride in (1, 2):
        paths = {}
        for length, factor in [('long', 5), ('short', 1)]:
            case = output / f'{length}-stride{stride}'
            case.mkdir()
            paths[length] = case
            for run in DURATION:
                inputs = write_case(source, case, run, DURATION[run]*factor, stride, store)
                phase = 'map' if run == 'mapping' else 'localize'
                args = [phase, case/run, '--config', PROFILE, '--out', case/phase]
                if phase == 'map':
                    args.extend(['--resolution', .08])
                else:
                    args.extend(['--map', paths['long']/'map'/'map.yaml', '--initial',
                                 *pose_at('inspection', 0)])
                stats = measure(args, case / f'{phase}.log')
                row = dict(length=length, stride=stride, phase=phase, **inputs, **stats)
                results.append(row)
                print(json.dumps(row), flush=True)
        for phase in ('map', 'localize'):
            comparisons.append(dict(stride=stride, phase=phase,
                                    **prefix_error(paths['short']/phase, paths['long']/phase)))
    report = dict(machine=hardware(), measurements=results, comparisons=comparisons)
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    benchmark(parser.parse_args().out)
