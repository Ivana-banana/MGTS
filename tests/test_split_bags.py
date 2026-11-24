import json
import subprocess
import sys
from contextlib import ExitStack

import pytest
import yaml
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from mgts_nav.bag import reader
from mgts_nav.demo import PROFILE, STORAGE, generate


def split_recording(source, target, storage):
    target.mkdir(parents=True)
    parts = [target.parent / f'{target.name}_{i}' for i in range(2)]
    store = get_typestore(Stores.ROS2_HUMBLE)
    with reader(source) as bag, ExitStack() as stack:
        writers = [stack.enter_context(Writer(part, version=8, storage_plugin=STORAGE[storage]))
                   for part in parts]
        connections = [{c.id: writer.add_connection(c.topic, c.msgtype, typestore=store)
                        for c in bag.connections} for writer in writers]
        midpoint = bag.start_time + bag.duration // 2
        for conn, timestamp, raw in bag.messages():
            index = int(timestamp >= midpoint)
            writers[index].write(connections[index][conn.id], timestamp, raw)

    metadata = yaml.safe_load((source / 'metadata.yaml').read_bytes())
    info = metadata['rosbag2_bagfile_information']
    files = []
    for part in parts:
        single = yaml.safe_load((part / 'metadata.yaml').read_bytes())
        entry, = single['rosbag2_bagfile_information']['files']
        assert entry['message_count'] > 0
        (part / entry['path']).rename(target / entry['path'])
        files.append(entry)
    info['files'] = files
    info['relative_file_paths'] = [entry['path'] for entry in files]
    (target / 'metadata.yaml').write_text(yaml.safe_dump(metadata), encoding='utf-8')
    return target / files[1]['path']


@pytest.mark.parametrize('storage', ['sqlite3', 'mcap'])
def test_solve_across_volume_boundary(tmp_path, storage):
    single = tmp_path / 'single'
    split = tmp_path / 'split'
    generate(single, storage=storage)
    split_recording(single / 'mapping', split / 'mapping', storage)
    second = split_recording(single / 'inspection', split / 'inspection', storage)
    start = json.loads((single / 'initial_pose.json').read_text())

    def run(data, output):
        return subprocess.run([sys.executable, '-m', 'mgts_nav.solve', str(data / 'mapping'),
                               str(data / 'inspection'), '--config', str(PROFILE),
                               '--initial', *map(str, start), '--out', str(output)],
                              capture_output=True, text=True)

    reference = tmp_path / 'reference'
    result = tmp_path / 'result'
    for data, output in [(single, reference), (split, result)]:
        process = run(data, output)
        assert process.returncode == 0, process.stderr
    for route, expected_count in [('map', 200), ('inspection', 100)]:
        filename = f'{route}/trajectory.csv'
        rows = (reference / filename).read_bytes()
        assert len(rows.splitlines()) == expected_count + 1
        assert (result / filename).read_bytes() == rows
    assert (result / 'inspection/continuous.csv').read_bytes() == (
        reference / 'inspection/continuous.csv').read_bytes()

    second.rename(second.with_suffix('.missing'))
    incomplete = tmp_path / 'incomplete'
    process = run(split, incomplete)
    assert process.returncode == 2
    assert second.name in process.stderr and 'missing' in process.stderr.lower()
    assert not incomplete.exists()
