import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

from mgts_nav.demo import PROFILE, generate
from tools.package_solution import build


def rename_volumes(data):
    for role, name in [('mapping', 'Измерения обхода.db3'), ('inspection', 'Инспекция офиса.db3')]:
        bag = data / role
        metadata = bag / 'metadata.yaml'
        document = yaml.safe_load(metadata.read_bytes())
        info = document['rosbag2_bagfile_information']
        original, = info['relative_file_paths']
        (bag / original).rename(bag / name)
        info['relative_file_paths'] = [name]
        info['files'][0]['path'] = name
        metadata.write_text(yaml.safe_dump(document, allow_unicode=True), encoding='utf-8')
        assert name.encode('utf-8') in metadata.read_bytes()


def test_utf8_volume_names_through_cli(tmp_path):
    data = tmp_path / 'recordings'
    generate(data)
    control = {}
    for mode, variant in [('0', 'ascii'), ('0', 'unicode'), ('1', 'unicode')]:
        if variant == 'unicode' and mode == '0':
            rename_volumes(data)

        env = dict(os.environ, PYTHONUTF8=mode)
        enc = json.loads(subprocess.check_output([sys.executable, '-c',
            'import json, locale, sys;print(json.dumps([locale.getpreferredencoding(False),'
            'sys.stdout.encoding, sys.stderr.encoding]))'], env=env))
        out = tmp_path / f'result-{variant}-{mode}'
        calls = {
            'inspect': ['mgts_nav', 'inspect', data/'mapping'],
            'profile': ['mgts_nav.profile', data/'mapping', '--verify-frames'],
            'solve': ['mgts_nav.solve', data/'mapping', data/'inspection', '--config', PROFILE, '--out', out],
        }
        for key, args in calls.items():
            proc = subprocess.run([sys.executable, '-m', *args], env=env, capture_output=True)
            print(variant, mode, *enc, key, proc.returncode)
            if proc.returncode:
                # Legacy locales may refuse the UTF-8 metadata; a corrected reader can also succeed.
                assert variant == 'unicode' and mode == '0' and enc[0].lower() != 'utf-8'
                detail = proc.stderr.decode(enc[2], errors='replace').lower()
                print(detail.strip())
                assert '-x utf8' in detail
                assert proc.returncode == 2 and any(s in detail for s in (
                    'decode', 'utf-8', 'missing', 'metadata', 'database')), detail
                assert not out.exists()
                continue
            if key == 'solve':
                result = [(out/path).read_bytes() for path in ('map/map.pgm', 'map/map.yaml',
                          'map/trajectory.csv', 'inspection/trajectory.csv', 'inspection/continuous.csv')]
            else:
                result = json.loads(proc.stdout.decode(enc[1]))
            if variant == 'ascii':
                control[key] = result
            else:
                assert result == control[key]


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / ".git").exists(),
                    reason="Building a historical ZIP needs the MGTS Git checkout")
def test_solve_children_inherit_utf8_mode(tmp_path):
    data = tmp_path / 'recordings'
    generate(data)
    rename_volumes(data)
    archive = tmp_path / 'runtime.zip'
    archive.write_bytes(build('HEAD')[1])
    probe_file = tmp_path / 'inspect_child_mode.py'
    probe_file.write_text("""
import sys
from pathlib import Path
import mgts_nav
mode, archive = sys.argv[1:]
assert sys.flags.utf8_mode == int(mode)
if archive:
    assert Path(archive) in Path(mgts_nav.__file__).parents
""", encoding='utf-8')
    probe_archive = tmp_path / 'probe.zip'
    probe_archive.write_bytes(archive.read_bytes())
    with zipfile.ZipFile(probe_archive, 'a') as bundle:
        bundle.write(probe_file, probe_file.name)
    probe = """
import subprocess, sys
location, mode = sys.argv[1:]
if location:
    sys.path.insert(0, location)
from mgts_nav.solve import module_command
subprocess.run(module_command('inspect_child_mode', mode, location), check=True)
"""
    for label, location in [('git', ''), ('zip', str(archive))]:
        isolation = ['-I'] if location else []
        for mode in (0, 1):
            subprocess.run([sys.executable, *isolation, '-X', f'utf8={mode}', '-c', probe,
                            str(probe_archive) if location else '', str(mode)],
                           env=dict(os.environ, PYTHONUTF8=str(1-mode), PYTHONPATH=str(tmp_path)), check=True)
        startup = ['-m', 'mgts_nav.solve']
        if location:
            startup = ['-c', "import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); "
                       "runpy.run_module('mgts_nav.solve', run_name='__main__', alter_sys=True)", location]
        subprocess.run([sys.executable, *isolation, '-X', 'utf8=1', *startup,
                        str(data/'mapping'), str(data/'inspection'), '--config', str(PROFILE),
                        '--out', str(tmp_path/label)], env=dict(os.environ, PYTHONUTF8='0'), check=True)
    for path in ('map/map.pgm', 'map/map.yaml', 'map/trajectory.csv', 'inspection/trajectory.csv',
                 'inspection/continuous.csv', 'baseline/trajectory.csv'):
        assert (tmp_path/'zip'/path).read_bytes() == (tmp_path/'git'/path).read_bytes()
