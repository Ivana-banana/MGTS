"""Exercise both ROS bag formats through the public offline CLI on the same measurements."""
import hashlib
import json
import subprocess
import sys

from rosbags.convert import convert
from rosbags.typesys import Stores, get_typestore

from mgts_nav.demo import PROFILE, generate


def invoke(module, *args):
    return subprocess.check_output([sys.executable, '-m', module, *map(str, args)], text=True)


def test_ros1_and_ros2_office_routes(tmp_path):
    ros2 = tmp_path / 'ros2'
    generate(ros2, seed=41)
    routes = ('mapping', 'inspection')
    for route in routes:
        target = tmp_path / f'{route}.bag'
        convert([ros2/route], target, dst_storage='sqlite3', dst_version=8, compress=None,
                compress_mode='file', default_typestore=get_typestore(Stores.ROS2_HUMBLE),
                typestore=get_typestore(Stores.ROS1_NOETIC), exclude_topics=(), include_topics=(),
                exclude_msgtypes=(), include_msgtypes=())
        with target.open('rb') as stream:
            assert stream.readline() == b'#ROSBAG V2.0\n'
        reports = []
        for bag in (ros2/route, target):
            inspected = json.loads(invoke('mgts_nav', 'inspect', bag))
            assert len(inspected) == 3
            assert all('header_start_s' in row and 'decode_error' not in row for row in inspected)
            sync = json.loads(invoke('mgts_nav.sync', bag, '--config', PROFILE, '--require-complete'))
            reports.append((sorted(inspected, key=lambda row: row['topic']), sync))
        assert reports[0] == reports[1]
        print(route, {r['topic']: r['messages'] for r in reports[0][0]}, reports[0][1]['coverage'])

    outputs = []
    for name, bags in [('ros2', [ros2/r for r in routes]),
                       ('ros1', [tmp_path/f'{r}.bag' for r in routes])]:
        out = tmp_path / f'{name}-solve'
        invoke('mgts_nav.solve', *bags, '--config', PROFILE, '--out', out)
        outputs.append(out)
    for artifact in ('map/map.pgm', 'map/map.yaml', 'map/trajectory.csv', 'inspection/trajectory.csv',
                     'inspection/summary.json', 'inspection/continuous.csv', 'baseline/trajectory.csv'):
        reference, candidate = [(out/artifact).read_bytes() for out in outputs]
        assert candidate == reference, artifact
        print(artifact, hashlib.sha256(candidate).hexdigest())
