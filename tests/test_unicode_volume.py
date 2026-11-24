import json
import os
import subprocess
import sys

import yaml

from mgts_nav.demo import PROFILE, generate


def test_utf8_volume_names_through_cli(tmp_path):
    data = tmp_path / 'recordings'
    generate(data)
    control = {}
    for mode, variant in [('0', 'ascii'), ('0', 'unicode'), ('1', 'unicode')]:
        if variant == 'unicode' and mode == '0':
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
