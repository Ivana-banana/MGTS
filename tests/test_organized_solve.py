"""Run solve on identical sweeps stored as packed and padded PointCloud2 messages."""
import csv
import hashlib
import json
import subprocess
import sys

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from mgts_nav.bag import reader, stamp
from mgts_nav.demo import generate
from tools.evaluate_deskew import POINT_DTYPE, sensor_profile, write_bag


def organize(source, target, lidar_topic):
    store = get_typestore(Stores.ROS2_HUMBLE)
    field = store.types['sensor_msgs/msg/PointField']
    # Physical order z, time, x, y. Descriptor order differs from both physical and XYZ order.
    offsets = {'z': 0, 'offset': 8, 'x': 12, 'y': 16}
    fields = [field(name, offsets[name], 5 if name == 'offset' else 7, 1)
              for name in ('y', 'offset', 'z', 'x')]
    times = []
    padding = 0
    with reader(source) as bag, Writer(target, version=8) as writer:
        outputs = {c.id: writer.add_connection(c.topic, c.msgtype, typestore=store) for c in bag.connections}
        for c, t, raw in bag.messages():
            if c.topic == lidar_topic:
                msg = bag.deserialize(raw, c.msgtype)
                assert msg.height == 1 and not msg.is_bigendian and msg.point_step == POINT_DTYPE.itemsize
                original = msg.data.view(POINT_DTYPE)
                h, w = 4, (len(original)+3)//4
                point_step = 24
                row_step = w*point_step + 13
                buf = np.full(h*row_step, 0xA5, dtype=np.uint8)
                valid = np.arange(h*w).reshape(h, w) < len(original)
                for name, offset in offsets.items():
                    is_time = name == 'offset'
                    plane = np.ndarray((h, w), '>i4' if is_time else '>f4', buffer=buf,
                                       offset=offset, strides=(row_step, point_step))
                    plane[valid] = original[name]
                    plane[~valid] = 0 if is_time else np.nan
                padding += int((~valid).sum())
                times.append(stamp(msg))
                msg.height, msg.width, msg.fields = h, w, fields
                msg.point_step, msg.row_step, msg.data = point_step, row_step, buf
                msg.is_bigendian, msg.is_dense = True, False
                raw = store.serialize_cdr(msg, c.msgtype)
            writer.write(outputs[c.id], t, raw)
    return times, padding


def test_solve_on_organized_big_endian_sweeps(tmp_path):
    original = tmp_path / 'original'
    generate(original, seed=41)
    cfg = sensor_profile()
    cfg['point_time'] = dict(field='offset', unit='ns')
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps(cfg))
    packed, organized = tmp_path/'packed', tmp_path/'organized'
    packed.mkdir()
    organized.mkdir()
    times = {}
    for run, seed in [('mapping', 41), ('inspection', 42)]:
        write_bag(original/run, packed/run, run, seed, cfg)
        times[run], padding = organize(packed/run, organized/run, cfg['lidar'])
        assert padding > 0
        print(f'{run}: scans={len(times[run])}, additional NaN cells={padding}')
    initial = json.loads((original/'initial_pose.json').read_bytes())
    outputs = []
    for case in [packed, organized]:
        out = case/'solution'
        subprocess.run([sys.executable, '-m', 'mgts_nav.solve', case/'mapping', case/'inspection',
                        '--config', profile, '--initial', *map(str, initial), '--out', out], check=True)
        outputs.append(out)
    for run, directory in [('mapping', 'map'), ('inspection', 'inspection')]:
        with (outputs[1]/directory/'trajectory.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert [float(row['t']) for row in rows] == times[run]
        assert any(row['status'] == 'matched' for row in rows)
    for name in ['map/map.pgm', 'map/map.yaml', 'map/trajectory.csv', 'inspection/trajectory.csv',
                 'map/summary.json', 'inspection/summary.json', 'inspection/continuous.csv']:
        a, b = [(out/name).read_bytes() for out in outputs]
        assert a == b, name
        print(name, hashlib.sha256(a).hexdigest())
