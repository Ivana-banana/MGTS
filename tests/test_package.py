import ast
import hashlib
import io
from pathlib import PurePosixPath
import re
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
        assert {name for name in names if name.startswith('tools/')} == {
            'tools/plot_quality.py', 'tools/evaluate_camera.py', 'tools/evaluate_deskew.py',
            'tools/benchmark_length.py',
        }
        assert all(name.startswith(('mgts_nav/', 'docs/', 'configs/', 'tools/', 'tests/test_'))
                   or name in ('README.md', 'requirements.txt', 'pytest.ini', 'Dockerfile', '.dockerignore')
                   for name in names)
        assert 'tests/test_package.py' not in names
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


def tracked(pattern):
    return set(subprocess.check_output(['git', 'ls-files', pattern], text=True).split())


def test_archive_runs_its_own_test_suite():
    # A module or test added to the repository must be listed, or the archive cannot run pytest.
    expected = tracked('mgts_nav/*.py') | tracked('tests/test_*.py')
    assert expected - set(FILES) == {'tests/test_package.py'}


def test_archived_documents_link_inside_archive():
    # README links to the team board and agent rules on purpose; they stay in the repository.
    internal = {'COORDINATION.md', 'AGENTS.md'}
    commit, payload = build('HEAD')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = set(archive.namelist())
        missing = []
        for name in names:
            if not name.endswith('.md'):
                continue
            text = archive.read(name).decode('utf-8-sig')
            for target in re.findall(r'\]\(([^)\s]+)\)', text):
                target = target.split('#', 1)[0]
                if not target or '://' in target or target in internal:
                    continue
                path = PurePosixPath(name).parent / target
                parts = []
                for part in path.parts:
                    if part == '..':
                        parts.pop()
                    elif part != '.':
                        parts.append(part)
                if '/'.join(parts) not in names:
                    missing.append(f'{name} -> {target}')
    assert missing == []
