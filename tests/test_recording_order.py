import csv
import json
import subprocess
import sys

from mgts_nav.demo import PROFILE, generate
from mgts_nav.solve import route_extent, same_recording, suspected_recording_swap


def route(path, points):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['t', 'x', 'y', 'yaw'])
        writer.writerows((i, x, y, 0) for i, (x, y) in enumerate(points))
    return path


def test_recording_order_requires_both_larger_length_and_larger_area(tmp_path):
    small = route(tmp_path / 'small.csv', [(0, 0), (1, 0), (1, 1)])
    large = route(tmp_path / 'large.csv', [(0, 0), (3, 0), (3, 3)])
    long_line = route(tmp_path / 'line.csv', [(0, 0), (5, 0), (10, 0)])
    assert suspected_recording_swap(large, small) is None
    order = suspected_recording_swap(small, large)
    assert order['suspected_swap']
    assert order['inspection'] == dict(duration_s=2, path_m=6, hull_area_m2=4.5)
    assert suspected_recording_swap(small, long_line) is None
    assert suspected_recording_swap(long_line, small) is None


def test_stationary_and_single_pose(tmp_path):
    point = route(tmp_path / 'single.csv', [(1, 2)])
    stationary = route(tmp_path / 'still.csv', [(1, 2)] * 4)
    assert route_extent(point) == dict(duration_s=0, path_m=0, hull_area_m2=0)
    assert suspected_recording_swap(point, stationary) is None


def test_solve_only_reports_suspected_swap(tmp_path):
    data = tmp_path / 'recordings'
    generate(data)
    for first, second in [('mapping', 'inspection'), ('inspection', 'mapping')]:
        output = tmp_path / first
        run = subprocess.run([sys.executable, '-m', 'mgts_nav.solve', data / first, data / second,
                              '--config', PROFILE, '--out', output], capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        summary = json.loads((output / 'inspection' / 'summary.json').read_bytes())
        if first == 'inspection':
            assert 'WARNING' in run.stderr and 'swapped' in run.stderr
            assert summary['recording_order']['suspected_swap']
        else:
            assert 'WARNING' not in run.stderr and 'recording_order' not in summary
            assert 'same_recording' not in summary


def test_same_recording_via_volume_and_directory(tmp_path):
    data = tmp_path / 'recordings'
    generate(data)
    directory = data / 'inspection'
    volume, = directory.glob('*.db3')
    assert same_recording(directory, directory)
    assert same_recording(volume, directory)
    assert not same_recording(data / 'mapping', volume)
    output = tmp_path / 'repeat'
    run = subprocess.run([sys.executable, '-m', 'mgts_nav.solve', directory, volume,
                          '--config', PROFILE, '--initial', '0', '0', '0', '--out', output],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert 'same recording' in run.stderr
    assert json.loads((output / 'inspection/summary.json').read_bytes())['same_recording']
