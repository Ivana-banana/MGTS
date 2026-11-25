import csv
import json
import subprocess
import sys

import numpy as np
import pytest
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from mgts_nav.bag import reader, stamp
from mgts_nav.demo import PROFILE, generate


@pytest.fixture(scope='module')
def office(tmp_path_factory):
    root = tmp_path_factory.mktemp('office') / 'data'
    generate(root)
    return root


def mapping_fixture(source, target, case):
    store = get_typestore(Stores.ROS2_HUMBLE)
    lidar = json.loads(PROFILE.read_bytes())['lidar']
    index = 0
    with reader(source) as bag, Writer(target, version=8) as writer:
        conns = {c.id: writer.add_connection(c.topic, c.msgtype, typestore=store) for c in bag.connections}
        for c, t, raw in bag.messages():
            if c.topic == lidar:
                index += 1
                if case == 'absent' or (case == 'single' and index > 1):
                    continue
                if case == 'sparse' or (case == 'partial' and index == 11):
                    msg = bag.deserialize(raw, c.msgtype)
                    msg.width = 3
                    msg.row_step = 3 * msg.point_step
                    msg.data = np.copy(msg.data[:msg.row_step])
                    raw = store.serialize_cdr(msg, c.msgtype)
                    dropped = stamp(msg)
                if case == 'duplicate_x' and index == 11:
                    msg = bag.deserialize(raw, c.msgtype)
                    msg.fields.append(next(f for f in msg.fields if f.name == 'x'))
                    raw = store.serialize_cdr(msg, c.msgtype)
            writer.write(conns[c.id], t, raw)
    if case == 'partial':
        return dropped


def commands(bag, office, profile):
    base = [sys.executable, '-m']
    return [[*base, 'mgts_nav', 'map', bag, '--config', profile],
            [*base, 'mgts_nav.solve', bag, office/'inspection', '--config', profile,
             '--inspection-config', PROFILE]]


@pytest.mark.parametrize('case,map_reason,solve_reason', [
    ('absent', 'scans=0', 'No messages on selected lidar'),
    ('height', '"outside_height_band": 200', '"outside_height_band": 200'),
    ('sparse', '"too_few_points": 200', '"too_few_points": 200'),
    ('single', 'Too few repeatedly observed occupied cells', 'Too few repeatedly observed occupied cells'),
    ('duplicate_x', "Duplicate PointCloud2 field 'x'", "Duplicate PointCloud2 field 'x'"),
])
def test_map_and_solve_leave_no_partial_output(office, tmp_path, case, map_reason, solve_reason):
    profile, bag = PROFILE, office/'mapping'
    if case == 'height':
        profile = tmp_path/'high.json'
        cfg = json.loads(PROFILE.read_bytes())
        cfg['height_band_m'] = [1000, 1001]
        profile.write_text(json.dumps(cfg))
    else:
        bag = tmp_path/'mapping'
        mapping_fixture(office/'mapping', bag, case)
    for i, (cmd, reason) in enumerate(zip(commands(bag, office, profile), [map_reason, solve_reason])):
        output = tmp_path/f'out-{i}'
        proc = subprocess.run([*cmd, '--out', output], capture_output=True, text=True)
        assert proc.returncode == 2 and reason in proc.stderr, proc.stderr
        assert not output.exists()


def test_partial_mapping_in_solve(office, tmp_path):
    bag = tmp_path/'mapping'
    dropped = mapping_fixture(office/'mapping', bag, 'partial')
    out = tmp_path/'solution'
    subprocess.run([*commands(bag, office, PROFILE)[1], '--out', out], check=True, capture_output=True)
    mapping = out/'map'
    summary = json.loads((mapping/'summary.json').read_bytes())
    assert summary['scan_rejections'] == dict(counts={'too_few_points': 1},
                                              timestamps_s={'too_few_points': [dropped]})
    with (mapping/'trajectory.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row['t']) for row in rows if row['status'] == 'prediction_only'] == [dropped]
