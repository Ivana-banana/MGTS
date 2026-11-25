"""Measure a fixed body-frame point displacement on the HEAP-1 revisit windows."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

from mgts_nav.bag import cloud_xyz, reader
from mgts_nav.grid import read_map
from mgts_nav.metrics import evaluate, read_csv
from tools.evaluate_micrometre import compare_tracks


def export(source, output, config, shift, count):
    store = get_typestore(Stores.ROS2_HUMBLE)
    rotation = Rotation.from_quat(config['base_from_lidar_xyzw'])
    displacement = np.array([shift, 0., 0.])
    sensor_delta = rotation.inv().apply(displacement)
    scans = 0
    with reader(source) as bag, Writer(output, version=8) as writer:
        targets = {c.id: writer.add_connection(c.topic, c.msgtype, typestore=store) for c in bag.connections}
        for c, t, raw in bag.messages():
            if c.topic == config['lidar']:
                msg = bag.deserialize(raw, c.msgtype)
                assert [f.name for f in msg.fields] == list('xyz') and msg.height == 1 and not msg.is_bigendian
                original = cloud_xyz(msg).astype('<f8')
                points = original + sensor_delta
                np.testing.assert_allclose(rotation.apply(points - original),
                                           np.broadcast_to(displacement, points.shape), atol=1e-13, rtol=0)
                for i, field in enumerate(msg.fields):
                    field.offset, field.datatype = 8*i, 8
                msg.point_step, msg.row_step, msg.data = 24, points.nbytes, points.view(np.uint8).ravel()
                raw = store.serialize_cdr(msg, c.msgtype)
                scans += 1
            writer.write(targets[c.id], t, raw)
    if scans != count:
        raise ValueError(f'Expected {count} clouds, found {scans}')


def nav(*args):
    return subprocess.run([sys.executable, '-X', 'utf8', '-m', 'mgts_nav', *map(str, args)],
                          capture_output=True, encoding='utf8')


def checked(result):
    if result.returncode:
        raise RuntimeError(result.stderr)


def experiment(data, root):
    root.mkdir(parents=True)
    profile = data/'profile.json'
    config = json.loads(profile.read_bytes())
    reports = {}
    bags = {}
    for name, shift in [('original', None), ('zero', 0.), ('shift', .000002)]:
        out = root/name
        out.mkdir()
        bags[name] = data if shift is None else out/'data'
        if shift is not None:
            bags[name].mkdir()
            for role, count in [('mapping', 40), ('inspection', 60)]:
                export(data/role, bags[name]/role, config, shift, count)
        checked(nav('map', bags[name]/'mapping', '--config', profile, '--resolution', '.08', '--out', out/'map'))
        result = nav('localize', bags[name]/'inspection', '--config', profile, '--map', out/'map/map.yaml',
                     '--start-search', 'delayed', '--out', out/'delayed')
        if result.returncode and not (result.returncode == 2 and 'Start search refused:' in result.stderr):
            checked(result)
        reports[name] = {'delayed_error': result.stderr.strip() if result.returncode else None}

    truth = {role: read_csv(data/f'{role}-reference.csv') for role in ('mapping', 'inspection')}
    initial = truth['inspection'][0]
    for name, row in reports.items():
        out = root/name
        checked(nav('localize', bags[name]/'inspection', '--config', profile, '--map', out/'map/map.yaml',
                    '--initial', initial.x, initial.y, initial.yaw, '--out', out/'known'))
        pixels, meta, xy = read_map(out/'map/map.yaml')
        cells = set(map(tuple, np.floor(xy/meta['resolution']).astype(int)))
        if name == 'original':
            original_cells = cells
            original_pixels, original_meta = pixels, meta
        row.update(occupied=len(cells), changed_cells=len(original_cells ^ cells))
        if pixels.shape == original_pixels.shape and meta == original_meta:
            row['changed_pixels'] = int(np.count_nonzero(pixels != original_pixels))
        for filename in ('map.yaml', 'map.pgm'):
            row[filename] = (out/'map'/filename).read_bytes() == (root/'original/map'/filename).read_bytes()
        for phase in ('map', 'delayed', 'known'):
            if phase == 'delayed' and row['delayed_error']:
                continue
            csv = out/phase/'trajectory.csv'
            summary = json.loads((out/phase/'summary.json').read_bytes())
            stats = evaluate(read_csv(csv), truth['mapping' if phase == 'map' else 'inspection'])
            stats.update(status=summary['status'], initial_pose=summary['initial_pose'])
            control = root/'original'/phase/'trajectory.csv'
            if control.exists():
                stats.update(compare_tracks(control, csv), identical=control.read_bytes() == csv.read_bytes())
            row[phase] = stats
    zero = reports['zero']
    assert zero['delayed_error'] == reports['original']['delayed_error']
    assert zero['map.yaml'] and zero['map.pgm'] and all(zero[p]['identical'] for p in ('map','known','delayed') if p in zero)
    (root/'report.json').write_text(json.dumps(reports, indent=2, allow_nan=False), encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    experiment(args.data, args.output)
