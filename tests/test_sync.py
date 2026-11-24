import json
import subprocess
import sys

import numpy as np
import pytest
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from mgts_nav.bag import load_config, reader, scans
from mgts_nav.clock_experiment import shift_imu
from mgts_nav.demo import PROFILE, generate
from mgts_nav.sync import _coverage, audit


@pytest.fixture(scope='module')
def recording(tmp_path_factory):
    root = tmp_path_factory.mktemp('sync') / 'data'
    generate(root)
    return root / 'inspection'


def assert_matches_loader(path, cfg, result):
    rows = list(scans(path, cfg))
    for role, count in [('odom', sum(row.odom is not None for row in rows)),
                        ('imu', sum(row.gyro_delta is not None for row in rows))]:
        assert result['coverage'][role]['covered'] == count


def test_shifted_header_clock_is_detected_and_restored(recording, tmp_path):
    cfg = load_config(PROFILE)
    normal = audit(recording, cfg)
    assert [normal['coverage'][k]['covered'] for k in ('odom', 'imu')] == [100, 99]
    assert_matches_loader(recording, cfg, normal)
    target = tmp_path / 'future'
    shift_imu(recording, target, cfg['imu'], 30_000_000_000)
    process = subprocess.run([sys.executable, '-m', 'mgts_nav.sync', str(target), '--config', str(PROFILE),
                              '--require-complete'], capture_output=True, text=True)
    assert process.returncode == 3, process.stderr
    result = json.loads(process.stdout)
    assert result['clocks']['imu']['header_range_s'] == [280., 290.]
    assert result['coverage']['imu']['missing']['before_start']['count'] == 99
    assert_matches_loader(target, cfg, result)
    fixed = dict(cfg, header_offset_s={'imu': -30.})
    result = audit(target, fixed)
    assert result['coverage'] == normal['coverage']
    assert result['clocks']['imu']['corrected_range_s'] == [250., 260.]
    assert_matches_loader(target, fixed, result)


def altered_imu(source, output, change):
    store = get_typestore(Stores.ROS2_HUMBLE)
    previous = None
    with reader(source) as bag, Writer(output, version=8) as writer:
        conns = {c.id: writer.add_connection(c.topic, c.msgtype, typestore=store) for c in bag.connections}
        for conn, received, raw in bag.messages():
            if conn.topic == '/demo/imu':
                msg = bag.deserialize(raw, conn.msgtype)
                header = msg.header.stamp
                ns = header.sec * 10**9 + header.nanosec
                if 254_000_000_000 < ns < 255_000_000_000:
                    if change == 'gap':
                        continue
                    if ns == 254_020_000_000:
                        header.sec, header.nanosec = divmod(previous - int(change == 'backwards'), 10**9)
                        raw = store.serialize_cdr(msg, conn.msgtype)
                previous = ns
            writer.write(conns[conn.id], received, raw)


def test_internal_gap_with_identical_outer_limits(recording, tmp_path):
    target = tmp_path / 'gap'
    altered_imu(recording, target, 'gap')
    cfg = load_config(PROFILE)
    report = audit(target, cfg)
    assert report['clocks']['imu']['header_range_s'] == [250., 260.]
    missing = report['coverage']['imu']['missing']['internal_gap']
    assert missing == dict(count=11, first=[253.935, 254.035], last=[254.935, 255.035])
    assert report['coverage']['imu']['status'] == 'partial'
    assert_matches_loader(target, cfg, report)


@pytest.mark.parametrize('bad', ['duplicate', 'backwards'])
def test_malformed_clock_reports_the_topic(recording, tmp_path, bad):
    target = tmp_path / 'bad'
    altered_imu(recording, target, bad)
    with pytest.raises(ValueError, match='/demo/imu: ' + bad):
        audit(target, load_config(PROFILE))


def test_disabled_and_empty_inputs_are_distinct(recording):
    report = audit(recording, dict(load_config(PROFILE), imu=None, odom=None))
    assert report['coverage'] == {'imu': {'status': 'disabled'}, 'odom': {'status': 'disabled'}}
    assert _coverage(np.array([]), [1.], [1.], .1)['missing']['no_messages']['count'] == 1
    assert _coverage(np.array([1.]), [], [], .1)['status'] == 'no_intervals'


def test_exact_samples_remain_usable_either_side_of_a_gap():
    clock = np.array([1., 2.])
    poses = _coverage(clock, [.5, 1., 1.5, 2., 2.5], [.5, 1., 1.5, 2., 2.5], .1)
    assert poses['covered'] == 2
    assert {k: v['count'] for k, v in poses['missing'].items()} == {
        'before_start': 1, 'internal_gap': 1, 'after_end': 1,
    }
    assert _coverage(clock, [.5], [2.5], .1)['missing']['outside_both_edges']['count'] == 1
