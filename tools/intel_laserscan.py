"""Replay the Intel #108 windows through ROS 2 LaserScan and compare with direct XYZ."""
import argparse
import csv
import json
from itertools import zip_longest
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
from rosbags.rosbag2 import Reader, Writer
from rosbags.typesys import Stores, get_typestore

from mgts_nav.metrics import read_csv
from tools.intel_lab import laser_rows
from tools.intel_laser_clock import ns
from tools.intel_order_odom import compare_epochs
from tools.laser_scan import convert


TOPIC = '/front_laser'


def make_scans(source, output, raw, store):
    scan_type = 'sensor_msgs/msg/LaserScan'
    lower = np.nextafter(np.float32(0), np.float32(1))
    upper = np.nextafter(np.float32(20), np.float32(0))
    with Reader(source) as bag, Writer(output, version=8) as writer:
        targets = {c.id: writer.add_connection(c.topic, scan_type if c.topic == TOPIC else c.msgtype,
                                               typestore=store) for c in bag.connections}
        for c, t, payload in bag.messages():
            target = targets[c.id]
            if c.topic == TOPIC:
                row = raw[t]
                stamp = store.types['builtin_interfaces/msg/Time'](*divmod(ns(row[-3]), 10**9))
                header = store.types['std_msgs/msg/Header'](stamp, 'front_laser')
                msg = store.types[scan_type](header, -np.pi/2, np.deg2rad(89), np.pi/180, 0., 0.,
                                             lower, upper, np.asarray(row[2:182], '<f4'), np.empty(0, '<f4'))
                payload = store.serialize_cdr(msg, scan_type)
            writer.write(target, t, payload)


def compare_clouds(direct, converted, raw, store):
    scans = beams = endpoints = odom = 0
    changed_cells = changed_counts = 0
    worst = 0.
    with Reader(direct) as a, Reader(converted) as b:
        for first, second in zip_longest(a.messages(), b.messages()):
            if first is None or second is None:
                raise ValueError('Message count changed')
            c, t, payload = first
            d, u, other = second
            if (c.topic, t, c.msgtype) != (d.topic, u, d.msgtype):
                raise ValueError('Message order, type or timestamp changed')
            if c.topic != TOPIC:
                if payload != other:
                    raise ValueError('Non-lidar payload changed')
                odom += 1
                continue
            old = store.deserialize_cdr(payload, c.msgtype)
            new = store.deserialize_cdr(other, d.msgtype)
            ranges = np.asarray(raw[t][2:182], float)
            endpoints += int(((ranges == 0) | (ranges == 20)).sum())
            expected = int(((0 < ranges) & (ranges < 20)).sum())
            if old.header != new.header or old.width != new.width or old.width != expected:
                raise ValueError('Header or beam count changed')
            xyz = old.data.view('<f4').reshape(-1, 3)
            xyzt = new.data.view('<f4').reshape(-1, 4)
            if np.any(xyzt[:, 3]):
                raise ValueError('Unexpected per-ray time')
            distances = np.linalg.norm(xyz.astype(float) - xyzt[:, :3], axis=1)
            cells = np.floor(xyz[:, :2].astype(float) / .06)
            other_cells = np.floor(xyzt[:, :2].astype(float) / .06)
            changed_cells += int(np.any(cells != other_cells, axis=1).sum())
            changed_counts += len(np.unique(cells, axis=0)) != len(np.unique(other_cells, axis=0))
            worst = max(worst, float(distances.max(initial=0.)))
            scans += 1
            beams += expected
    return dict(scans=scans, kept=beams, at_zero_or_twenty=endpoints, odom=odom, max_xyz_m=worst,
                changed_voxel_beams=changed_cells, changed_voxel_count_scans=changed_counts)


def trajectory(a, b):
    with a.open(encoding='utf-8', newline='') as af, b.open(encoding='utf-8', newline='') as bf:
        rows, other = list(csv.DictReader(af)), list(csv.DictReader(bf))
    if [r['t'] for r in rows] != [r['t'] for r in other]:
        raise ValueError('Trajectory timestamps changed')
    delta = np.array([[float(r[k])-float(s[k]) for k in ('x', 'y', 'yaw')] for r, s in zip(rows, other)])
    xy = np.linalg.norm(delta[:, :2], axis=1)
    return dict(status_changes=sum(r['status'] != s['status'] for r, s in zip(rows, other)),
                rms_xy_m=float(np.sqrt(np.mean(xy**2))), max_xy_m=float(xy.max()),
                max_yaw_deg=float(np.rad2deg(np.abs(np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2])))).max()))


def run(log, data, out):
    out.mkdir(parents=True, exist_ok=False)
    store = get_typestore(Stores.ROS2_HUMBLE)
    raw = {ns(row[-3]): row for row in laser_rows(log)}
    report = {}
    for role in ('mapping', 'inspection'):
        scan = out / (role + '-scan')
        make_scans(data / role, scan, raw, store)
        convert(scan, TOPIC, out / role)
        report[role] = compare_clouds(data / role, out / role, raw, store)
    cli = [sys.executable, '-m', 'mgts_nav']
    for label, inputs in [('direct', data), ('converted', out)]:
        base = out / label
        base.mkdir()
        common = ['--config', data / 'profile.json']
        subprocess.run([*cli, 'map', inputs / 'mapping', *common, '--resolution', '.08',
                        '--out', base / 'map'], check=True, stdout=subprocess.DEVNULL)
        subprocess.run([*cli, 'localize', inputs / 'inspection', *common, '--map', base / 'map/map.yaml',
                        '--initial-file', data / 'external-start.json', '--out', base / 'known'],
                       check=True, stdout=subprocess.DEVNULL)
    a, b = out/'direct', out/'converted'
    fixed = out / 'converted-on-direct'
    subprocess.run([*cli, 'localize', out/'inspection', '--config', data/'profile.json',
                    '--map', a/'map/map.yaml', '--initial-file', data/'external-start.json',
                    '--out', fixed], check=True, stdout=subprocess.DEVNULL)
    report['fixed_map'] = trajectory(a/'known/trajectory.csv', fixed/'trajectory.csv')
    for name in ('map', 'known'):
        report[name] = trajectory(a/name/'trajectory.csv', b/name/'trajectory.csv')
        role = 'mapping' if name == 'map' else 'inspection'
        reference = read_csv(data / f'{role}-external.csv')
        for label, directory in [('direct', a), ('converted', b)]:
            report[name][label] = json.loads((directory/name/'summary.json').read_bytes())['status']
            report[name][label + '_rbpf'] = compare_epochs(directory/name/'trajectory.csv', reference)
    report['same_map_yaml'] = (a/'map/map.yaml').read_bytes() == (b/'map/map.yaml').read_bytes()
    with Image.open(a/'map/map.pgm') as first, Image.open(b/'map/map.pgm') as second:
        report['map_sizes'] = [first.size, second.size]
        if report['same_map_yaml'] and first.size == second.size:
            report['changed_pixels'] = int(np.count_nonzero(np.asarray(first) != np.asarray(second)))
    text = json.dumps(report, indent=2, allow_nan=False)
    (out/'comparison.json').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('log', 'data', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.log, args.data, args.output)
