import ast
import hashlib
import json
import os
import subprocess
import sys
import zipfile

import pytest

from tools.package_solution import FILES, build, verify


ZIP_CLI = """
import importlib.util, runpy, sys
from pathlib import Path
from zipimport import zipimporter
archive, module = sys.argv[1:3]
sys.path.insert(0, archive)
spec = importlib.util.find_spec(module)
assert isinstance(spec.loader, zipimporter) and Path(spec.loader.archive) == Path(archive)
import mgts_nav
assert Path(archive) in Path(mgts_nav.__file__).parents
sys.argv = [spec.origin, *sys.argv[3:]]
runpy.run_module(module, run_name='__main__')
"""


def test_diagnostics_from_zip(tmp_path):
    from mgts_nav.demo import PROFILE, generate

    _, payload = build('HEAD')
    archive = tmp_path / 'runtime.zip'
    archive.write_bytes(payload)
    with zipfile.ZipFile(archive) as source:
        for guide in ('delayed-start', 'delayed-start-controls', 'keyframe-start', 'coarse-start',
                      'obstacles', 'solve-manifest', 'interior-loop', 'heap1-velocity-prior'):
            assert source.getinfo(f'docs/{guide}.md').file_size > 0
    prefix = [sys.executable, '-I', '-c', ZIP_CLI, str(archive)]
    for module in ('delayed_start', 'obstacles', 'solve_audit', 'solve_manifest', 'interior_loop'):
        help_text = subprocess.check_output([*prefix, f'mgts_nav.{module}', '--help'], text=True)
        assert 'usage:' in help_text

    data = tmp_path / 'office'
    generate(data)
    out = tmp_path / 'solve'
    subprocess.run([sys.executable, '-m', 'mgts_nav.solve', str(data/'mapping'), str(data/'inspection'),
                    '--config', str(PROFILE), '--out', str(out)], check=True)
    # Detect a nested interpreter that accidentally imports the checkout or a user override.
    marker = tmp_path / 'unisolated-child'
    (tmp_path/'sitecustomize.py').write_text(
        f'from pathlib import Path\nPath({str(marker)!r}).touch()\n', encoding='utf-8')
    packed_out = tmp_path / 'packed-solve'
    subprocess.run([*prefix, 'mgts_nav.solve', str(data/'mapping'), str(data/'inspection'),
                    '--config', str(PROFILE), '--out', str(packed_out)], check=True,
                   env=dict(os.environ, PYTHONPATH=str(tmp_path)))
    assert not marker.exists()
    for path in ('map/map.pgm', 'map/map.yaml', 'map/trajectory.csv', 'inspection/trajectory.csv',
                 'inspection/continuous.csv', 'baseline/trajectory.csv'):
        assert (packed_out/path).read_bytes() == (out/path).read_bytes(), path
    args = ['mgts_nav.interior_loop', str(data/'mapping'), str(out/'map/trajectory.csv'),
            '--config', str(out/'profiles/mapping.json')]
    packaged = subprocess.check_output([*prefix, *args])
    assert packaged == subprocess.check_output([sys.executable, '-m', *args])
    report = json.loads(packaged)
    assert report['counts']['scans'] == 200 and report['candidates']
    print('ZIP and checkout:', report['counts']['scans'], 'scans,', len(report['candidates']),
          'pairs, SHA-256', hashlib.sha256(packaged).hexdigest())


def test_reproducible_committed_runtime(tmp_path):
    commit, payload = build('HEAD')
    assert build(commit) == (commit, payload)
    archive = tmp_path / 'candidate.zip'
    archive.write_bytes(payload)
    verify(archive, hashlib.sha256(payload).hexdigest())
    with zipfile.ZipFile(archive) as result:
        names = result.namelist()
        assert tuple(names) == FILES
        assert {'mgts_nav/image_pairs.py', 'docs/image-pairs.md'} <= set(names)
        assert {name for name in names if name.startswith('configs/')} == {
            'configs/example.json', 'configs/inspection.rviz', 'configs/demo-zones.json',
        }
        assert {name for name in names if name.startswith('tools/')} == {
            'tools/plot_quality.py', 'tools/evaluate_camera.py', 'tools/evaluate_deskew.py',
            'tools/native_grid.cpp', 'tools/benchmark_native_grid.py',
            'tools/plot_heap1_revisit.py',
        }
        assert all(name.startswith(('mgts_nav/', 'docs/', 'configs/', 'tools/'))
                   or name in ('README.md', 'requirements.txt') for name in names)
        assert not any(name.endswith(('.bag', '.mcap', '.db3')) for name in names)
        for name in names:
            if name.startswith('mgts_nav/'):
                module = ast.parse(result.read(name), filename=name)
                for node in ast.walk(module):
                    if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                        assert f'mgts_nav/{node.module}.py' in names
        original = subprocess.check_output(['git', 'show', commit + ':configs/example.json'])
        assert result.read('configs/example.json') == original
    archive.write_bytes(payload[:-1])
    with pytest.raises(ValueError, match='SHA-256'):
        verify(archive, hashlib.sha256(payload).hexdigest())


def test_cli_preserves_existing_destination(tmp_path):
    archive = tmp_path / 'existing.zip'
    archive.write_bytes(b'keep')
    result = subprocess.run([sys.executable, 'tools/package_solution.py', str(archive)],
                            capture_output=True)
    assert result.returncode == 2
    assert archive.read_bytes() == b'keep'


@pytest.mark.parametrize('revision,digest', [
    ('3aef5eaf1f4a01ff87ffd0393eaac1fb4edfa986', '372759b04cc3b8a0ba30cd5a2d381dc8c5799c2486980d6effcc0a8fd023d941'),
    ('9bab80b7b989289081e46c72066a9d08dd4d8efa', 'c967de77b0b6cf2a2060334b843e035a610e022c5f7c07f45365804c361dec82'),
])
def test_historical_package(tmp_path, revision, digest):
    archive = tmp_path / 'historical.zip'
    cmd = [sys.executable, 'tools/package_solution.py', str(archive), '--revision', revision]
    subprocess.run(cmd, check=True)
    original = archive.read_bytes()
    assert hashlib.sha256(original).hexdigest() == digest
    subprocess.run([*cmd, '--verify'], check=True)
    archive.write_bytes(original[:100] + bytes([original[100] ^ 1]) + original[101:])
    failed = subprocess.run([*cmd, '--verify'], capture_output=True)
    assert failed.returncode == 2 and b'SHA-256' in failed.stderr
