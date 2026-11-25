"""Compare a fixed two-micrometre shift against the synthetic office truth (issue 144)."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from mgts_nav.bag import reader
from mgts_nav.demo import PROFILE, WALLS, generate
from mgts_nav.grid import read_map
from mgts_nav.map_eval import evaluate_map
from tools.evaluate_lidar_extrinsics import run, scores


def rewrite(source, output, shift):
    store = get_typestore(Stores.ROS2_HUMBLE)
    smallest, largest = np.inf, -np.inf
    with reader(source) as bag, Writer(output, version=8) as writer:
        connections = {c.id: writer.add_connection(c.topic, c.msgtype, typestore=store)
                       for c in bag.connections}
        for c, t, raw in bag.messages():
            if c.msgtype.endswith('/PointCloud2'):
                msg = bag.deserialize(raw, c.msgtype)
                assert [(f.name, f.offset, f.datatype) for f in msg.fields] == [
                    ('x', 0, 7), ('y', 4, 7), ('z', 8, 7)]
                assert msg.point_step == 12 and not msg.is_bigendian and msg.height == 1
                xyz = np.frombuffer(msg.data, '<f4').reshape(-1, 3).copy()
                original = xyz[:, 0].astype(float)
                xyz[:, 0] = original + shift
                delta = xyz[:, 0] - original
                smallest = min(smallest, delta.min())
                largest = max(largest, delta.max())
                msg.data = xyz.view(np.uint8).ravel()
                raw = store.serialize_cdr(msg, c.msgtype)
            writer.write(connections[c.id], t, raw)
    return [float(smallest), float(largest)]


def compare_tracks(first, second):
    def rows(path):
        with path.open(newline='') as stream:
            return list(csv.DictReader(stream))
    a, b = rows(first), rows(second)
    if [r['t'] for r in a] != [r['t'] for r in b]:
        raise ValueError('Trajectory clocks differ')
    distances = [np.hypot(float(p['x']) - float(q['x']), float(p['y']) - float(q['y']))
                 for p, q in zip(a, b)]
    return dict(max_xy_m=max(distances), status_changes=sum(p['status'] != q['status'] for p, q in zip(a, b)))


def occupied(directory):
    _, meta, xy = read_map(directory / 'map.yaml')
    cells = set(map(tuple, np.floor(xy / meta['resolution']).astype(int)))
    return cells, evaluate_map(xy, WALLS, meta['resolution'], .12)


def experiment(root):
    root.mkdir(parents=True)
    no_odom = root / 'no-odom.json'
    profile = json.loads(PROFILE.read_bytes())
    profile['odom'] = None
    no_odom.write_text(json.dumps(profile), encoding='utf8')
    results = []
    for seed in (41, 73):
        case = root / str(seed)
        source = case / 'original'
        generate(source, seed=seed)
        for variant, shift in [('original', None), ('zero', 0.), ('shift', .000002)]:
            data = case / variant
            changes = {}
            if shift is not None:
                data.mkdir()
                for role in ('mapping', 'inspection'):
                    changes[role] = rewrite(source/role, data/role, shift)
            for mode, config in [('odom', PROFILE), ('no-odom', no_odom)]:
                output = case / f'{variant}-{mode}'
                output.mkdir()
                run('map', data/'mapping', '--config', config, '--resolution', '.08', '--out', output/'map')
                run('localize', data/'inspection', '--config', config, '--map', output/'map/map.yaml',
                    '--initial-file', source/'initial_pose.json', '--out', output/'inspection')
                control = case / f'original-{mode}'
                cells, geometry = occupied(output/'map')
                base_cells, _ = occupied(control/'map')
                row = dict(seed=seed, mode=mode, variant=variant, displacement_range_m=changes,
                           geometry=geometry, occupied_symdiff=len(base_cells ^ cells))
                for phase, role in [('map', 'mapping'), ('inspection', 'inspection')]:
                    path = Path(phase) / 'trajectory.csv'
                    row[phase] = scores(output/phase, source/f'{role}_truth.csv')
                    row[phase].update(compare_tracks(control/path, output/path))
                if variant == 'zero':
                    for path in ('map/map.yaml', 'map/map.pgm', 'map/trajectory.csv', 'inspection/trajectory.csv'):
                        if (control/path).read_bytes() != (output/path).read_bytes():
                            raise ValueError(f'Zero control differs: {seed}/{mode}/{path}')
                results.append(row)
                print(seed, mode, variant, flush=True)
    (root/'report.json').write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    experiment(parser.parse_args().output)
