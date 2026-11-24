import ast
import hashlib
import subprocess
import sys
import zipfile

import pytest

from tools.package_solution import FILES, build, verify


def test_reproducible_committed_runtime(tmp_path):
    commit, payload = build('HEAD')
    assert build(commit) == (commit, payload)
    archive = tmp_path / 'candidate.zip'
    archive.write_bytes(payload)
    verify(archive, hashlib.sha256(payload).hexdigest())
    with zipfile.ZipFile(archive) as result:
        names = result.namelist()
        assert tuple(names) == FILES
        assert {name for name in names if name.startswith('configs/')} == {
            'configs/example.json', 'configs/inspection.rviz', 'configs/demo-zones.json',
        }
        assert all(name.startswith(('mgts_nav/', 'docs/', 'configs/'))
                   or name in ('README.md', 'requirements.txt', 'tools/plot_quality.py') for name in names)
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
