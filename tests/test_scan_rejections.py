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
from tools.evaluate_deskew import sensor_profile, write_bag


def inject_bad_clouds(source, target, lidar_topic):
    """Alter five spaced-out messages in the demo's compact XYZ cloud."""
    cases = {10: 'empty_input', 20: 'outside_height_band', 30: 'icp_rejected',
             40: 'no_finite_points', 50: 'too_few_points'}
    expected = {}
    store = get_typestore(Stores.ROS2_HUMBLE)
    index = 0
    with reader(source) as bag, Writer(target, version=8) as writer:
        outputs = {c.id: writer.add_connection(c.topic, c.msgtype, typestore=store)
                   for c in bag.connections}
        for c, t, raw in bag.messages():
            if c.topic == lidar_topic:
                reason = cases.get(index)
                index += 1
                if reason:
                    msg = bag.deserialize(raw, c.msgtype)
                    expected[reason] = [stamp(msg)]
                    xyz = msg.data.view('<f4').reshape(-1, 3).copy()
                    if reason == 'empty_input':
                        xyz = xyz[:0]
                    elif reason == 'outside_height_band':
                        xyz[:, 2] += 100.
                    elif reason == 'icp_rejected':
                        xyz[:, 0] += 100.
                    elif reason == 'no_finite_points':
                        xyz[:, 0] = np.nan
                    else:
                        xyz = xyz[:5]
                    msg.width, msg.row_step = len(xyz), xyz.nbytes
                    msg.data = xyz.view(np.uint8).ravel()
                    raw = store.serialize_cdr(msg, c.msgtype)
            writer.write(outputs[c.id], t, raw)
    return expected


def test_reasons_from_bag_through_localize(tmp_path):
    data = tmp_path / 'demo'
    generate(data, seed=41)
    cli = [sys.executable, '-m', 'mgts_nav']
    subprocess.run([*cli, 'map', data/'mapping', '--config', PROFILE,
                    '--out', tmp_path/'map'], check=True)
    profile = sensor_profile()
    bad = tmp_path / 'bad'
    expected = inject_bad_clouds(data/'inspection', bad, profile['lidar'])
    gap = tmp_path / 'gap'
    write_bag(data/'inspection', gap, 'inspection', 41, profile, odom_gap=True)
    deskew_config = tmp_path / 'deskew.json'
    deskew_config.write_text(json.dumps(dict(profile, point_time=dict(field='offset', unit='ns'))))
    for label, source, config, times in [
        ('clean', data/'inspection', PROFILE, {}),
        ('altered', bad, PROFILE, expected),
        ('deskew', gap, deskew_config, {'deskew_odom_unavailable': [254.035+i/10 for i in range(6)]}),
    ]:
        out = tmp_path / f'result-{label}'
        subprocess.run([*cli, 'localize', source, '--config', config, '--map', tmp_path/'map/map.yaml',
                        '--initial-file', data/'initial_pose.json', '--out', out], check=True)
        summary = json.loads((out/'summary.json').read_text())
        report = summary['scan_rejections']
        assert report['counts'] == {reason: len(ts) for reason, ts in times.items()}
        assert report['timestamps_s'].keys() == times.keys()
        for reason, ts in times.items():
            assert report['timestamps_s'][reason] == pytest.approx(ts, rel=0, abs=1e-8)
        with (out/'trajectory.csv').open(newline='') as stream:
            rejected = [float(row['t']) for row in csv.DictReader(stream)
                        if row['status'] == 'prediction_only']
        assert sorted(t for ts in report['timestamps_s'].values() for t in ts) == rejected
