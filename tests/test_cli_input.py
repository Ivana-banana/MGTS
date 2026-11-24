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
    ['mgts_nav.observability', '--bag', '{bag}', '--config', PROFILE],
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


@pytest.mark.parametrize('text, phrase', [('{"x": 1, "y": 2, "yaw": 0}', 'must hold a JSON array [x, y, yaw]'),
                                          ('[1, 2]', 'must hold a JSON array [x, y, yaw]'),
                                          ('[1, 2, 0,]', 'invalid JSON in --initial-file')])
def test_initial_file_errors_name_the_file(demo, tmp_path, text, phrase):
    start = tmp_path / 'start.json'
    start.write_text(text)
    refused(localize(demo, tmp_path, '--initial-file', start), str(start), phrase)


def broken_profile(tmp_path):
    profile = tmp_path / 'trailing.json'
    profile.write_text(PROFILE.read_text().rstrip().rstrip('}').rstrip() + ',\n}\n')
    return profile


def test_profile_json_error_names_the_file(demo, tmp_path):
    profile = broken_profile(tmp_path)
    run = cli('mgts_nav', 'map', demo / 'mapping', '--config', profile, '--out', tmp_path / 'out')
    refused(run, f'{profile}: invalid JSON in sensor profile')


def test_solve_names_the_broken_inspection_profile(demo, tmp_path):
    profile = broken_profile(tmp_path)
    run = cli('mgts_nav.solve', demo / 'mapping', demo / 'inspection', '--config', PROFILE,
              '--inspection-config', profile, '--out', tmp_path / 'out')
    refused(run, f'{profile}: invalid JSON in sensor profile')


@pytest.mark.parametrize('values, phrase', [(['-1', '0.55'], "got '--out'"), (['1,5', '0', '0'], 'decimal separator')])
def test_initial_explains_three_values_in_radians(demo, tmp_path, values, phrase):
    refused(localize(demo, tmp_path, '--initial', *values), phrase, 'X Y YAW (metres, metres, radians)')


def test_initial_help_names_units():
    for module in (['mgts_nav', 'localize'], ['mgts_nav.solve']):
        run = cli(*module, '--help')
        assert 'YAW in' in run.stdout and 'radians' in run.stdout


def test_continuous_short_row_names_the_line(demo, tmp_path):
    scans = tmp_path / 'scans.csv'
    scans.write_text('t,x,y,yaw,status\n1.0,0,0,0,matched\n2.0,0,0\n')
    run = cli('mgts_nav.continuous', demo / 'inspection', scans, tmp_path / 'dense.csv', '--config', PROFILE)
    refused(run, f'{scans}:3: invalid number')


@pytest.mark.parametrize('command', [
    ['mgts_nav.loop', '{mapping}', '{missing}', '--config', PROFILE],
    ['mgts_nav.loop', '{mapping}', '{text}', '--config', PROFILE],
    ['mgts_nav.map_eval', '{missing}'],
])
def test_auxiliary_clis_report_input_errors(demo, tmp_path, command):
    text = tmp_path / 'words.csv'
    text.write_text('t,x,y,yaw\na,b,c,d\n')
    names = dict(mapping=demo / 'mapping', missing=tmp_path / 'missing.csv', text=text)
    refused(cli(*[str(a).format(**names) for a in command]))
