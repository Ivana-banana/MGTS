"""Typical input mistakes end with a one-line message and exit code 2, not a traceback."""
import subprocess
import sys

import pytest

from mgts_nav.demo import PROFILE, generate


def cli(*args):
    return subprocess.run([sys.executable, '-m', *map(str, args)], capture_output=True, text=True)


def refused(run, *phrases):
    assert run.returncode == 2, run.stderr
    assert 'Traceback' not in run.stderr
    for phrase in phrases:
        assert phrase in run.stderr, run.stderr


@pytest.fixture(scope='module')
def demo(tmp_path_factory):
    data = tmp_path_factory.mktemp('demo') / 'data'
    generate(data)
    subprocess.run([sys.executable, '-m', 'mgts_nav', 'map', str(data / 'mapping'), '--config', str(PROFILE),
                    '--out', str(data / 'map')], check=True, capture_output=True)
    return data


@pytest.mark.parametrize('command', [
    ['mgts_nav', 'inspect', '{bag}'],
    ['mgts_nav', 'map', '{bag}', '--config', PROFILE, '--out', '{out}'],
    ['mgts_nav', 'localize', '{bag}', '--config', PROFILE, '--map', '{map}', '--initial', 0, 0, 0,
     '--out', '{out}'],
    ['mgts_nav', 'baseline', '{bag}', '--config', PROFILE, '--initial', 0, 0, 0, '--out', '{out}'],
    ['mgts_nav.calibration', '{bag}', '--base', 'a', '--lidar', 'b', '--imu', 'c'],
    ['mgts_nav.profile', '{bag}'],
])
def test_unreadable_recording_is_reported(demo, tmp_path, command):
    bag = tmp_path / 'notabag.txt'
    bag.write_text('not a bag')
    args = [str(a).format(bag=bag, out=tmp_path / 'out', map=demo / 'map' / 'map.yaml') for a in command]
    refused(cli(*args), 'cannot open recording')
    assert not (tmp_path / 'out').exists()


def test_metadata_file_instead_of_recording_directory(demo, tmp_path):
    run = cli('mgts_nav', 'map', demo / 'mapping' / 'metadata.yaml', '--config', PROFILE,
              '--out', tmp_path / 'out')
    refused(run, f'Pass the recording directory {demo / "mapping"}')


def localize(demo, tmp_path, *options, map_path=None):
    return cli('mgts_nav', 'localize', demo / 'inspection', '--config', PROFILE,
               '--map', map_path or demo / 'map' / 'map.yaml', *options, '--out', tmp_path / 'out')


@pytest.mark.parametrize('name, phrase', [('map.pgm', 'needs map.yaml'), ('', 'not a directory'),
                                          ('bad.yaml', 'not a map.yaml')])
def test_map_option_names_map_yaml(demo, tmp_path, name, phrase):
    (demo / 'map' / 'bad.yaml').write_text('just text')
    refused(localize(demo, tmp_path, '--initial', 0, 0, 0, map_path=demo / 'map' / name), phrase)
