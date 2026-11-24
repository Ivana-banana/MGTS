"""Build or verify the MGTS runtime ZIP from an immutable Git revision."""
import argparse
import ast
import hashlib
import io
from pathlib import Path
import subprocess
import os
from tempfile import NamedTemporaryFile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    '__init__', '__main__', 'bag', 'benchmark', 'calibration', 'camera_demo', 'continuous', 'demo',
    'coarse_start', 'delayed_start', 'deskew', 'grid', 'image_pairs', 'initialization', 'interior_loop',
    'leveling', 'loop', 'map_eval', 'metrics', 'obstacles', 'solve_manifest',
    'multiscan', 'native_grid', 'observability', 'plan_alignment', 'pose_graph', 'profile', 'quality', 'recovery', 'rehearse',
    'ros_export', 'slam', 'solve', 'start_confirmation', 'stress', 'sync', 'vision', 'visual_fusion', 'zones',
)
FILES = tuple(sorted((
    'README.md', 'requirements.txt', 'configs/example.json', 'configs/inspection.rviz', 'configs/demo-zones.json',
    'docs/ambiguity.md', 'docs/corridor.md', 'docs/demo-results.md', 'docs/demo-results.png',
    'tools/plot_quality.py', 'docs/quality-clean.png', 'docs/quality-gap.png',
    'tools/evaluate_camera.py', 'docs/camera-experiment.md', 'docs/image-pairs.md',
    'tools/evaluate_deskew.py', 'docs/deskew-experiment.md',
    'docs/coarse-start.md', 'docs/delayed-start.md', 'docs/delayed-start-controls.md',
    'docs/keyframe-start.md', 'docs/obstacles.md', 'docs/solve-manifest.md', 'docs/interior-loop.md',
    'tools/native_grid.cpp', 'tools/benchmark_native_grid.py', 'docs/native-grid.md',
    'docs/plan-alignment.md', 'docs/plan-landmarks.json', 'docs/office-plan.png', 'docs/plan-route.png',
    'docs/intake.md', 'docs/public-data.md', 'docs/report.md', 'docs/runtime.md',
    'docs/scan-rejections.md', 'docs/sensors.md', 'docs/start-confirmation.md', 'docs/zones.md', 'docs/demo-zone-visits.csv',
    *(f'mgts_nav/{name}.py' for name in MODULES),
)))


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def revision_files(commit):
    """Read the selected packager's declarative inventory without executing revision code."""
    tree = ast.parse(git('show', commit + ':tools/package_solution.py'))
    declarations = {node.targets[0].id: node.value for node in tree.body
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    expr = declarations['FILES']
    if not (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == 'tuple'
            and len(expr.args) == 1 and not expr.keywords):
        raise ValueError('Unsupported FILES declaration in ' + commit)
    order = expr.args[0]
    if not (isinstance(order, ast.Call) and isinstance(order.func, ast.Name) and order.func.id == 'sorted'
            and len(order.args) == 1 and not order.keywords and isinstance(order.args[0], ast.Tuple)):
        raise ValueError('Expected sorted literal inventory in ' + commit)
    paths = []
    expansions = {ast.dump(ast.parse(f"(f'{prefix}{{name}}.py' for name in {group})", mode='eval').body):
                  (group, prefix) for group, prefix in [('MODULES', 'mgts_nav/'), ('TESTS', 'tests/test_')]}
    for item in order.args[0].elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            paths.append(item.value)
        elif isinstance(item, ast.Starred) and ast.dump(item.value) in expansions:
            group, prefix = expansions[ast.dump(item.value)]
            names = ast.literal_eval(declarations[group])
            if not isinstance(names, tuple) or not all(isinstance(name, str) for name in names):
                raise ValueError('Expected literal module names in ' + group)
            paths.extend(prefix + name + '.py' for name in names)
        else:
            raise ValueError('Unsupported inventory entry in ' + commit)
    return tuple(sorted(paths))


def build(revision):
    commit = git('rev-parse', '--verify', '--end-of-options', revision + '^{commit}').decode().strip()
    files = revision_files(commit)
    entries = {}
    for record in git('ls-tree', '-r', '-z', commit, '--', *files).split(b'\0'):
        if record:
            header, name = record.split(b'\t', 1)
            mode, kind, oid = header.split()
            if mode not in (b'100644', b'100755') or kind != b'blob':
                raise ValueError(f'Expected a regular file: {name!r}')
            entries[name.decode()] = oid
    # Request immutable blobs directly so checkout filters cannot convert line endings.
    request = b''.join(entries[path] + b'\n' for path in files)
    blobs = io.BytesIO(subprocess.check_output(['git', 'cat-file', '--batch'], input=request, cwd=ROOT))
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w') as archive:
        for path in files:
            oid, kind, size = blobs.readline().split()
            if oid != entries[path] or kind != b'blob':
                raise ValueError(f'Unexpected Git object for {path}')
            content = blobs.read(int(size))
            if blobs.read(1) != b'\n':
                raise ValueError(f'Incomplete Git response for {path}')
            # Fixed ZIP metadata and stored bytes give the same digest across hosts.
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    return commit, output.getvalue()


def verify(path, expected_digest, files=FILES):
    with path.open('rb') as stream:
        actual = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            actual.update(block)
        if actual.hexdigest() != expected_digest:
            raise ValueError('SHA-256 differs from the selected revision')
        with zipfile.ZipFile(stream) as archive:
            if tuple(archive.namelist()) != files or archive.testzip() is not None:
                raise ValueError('ZIP membership or CRC mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--verify', action='store_true', help='Check an existing ZIP')
    args = parser.parse_args()
    try:
        commit, payload = build(args.revision)
        with zipfile.ZipFile(io.BytesIO(payload)) as expected:
            files = tuple(expected.namelist())
        digest = hashlib.sha256(payload).hexdigest()
        if not args.verify:
            args.archive.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=args.archive.parent, delete=False) as stage:
                staged_path = Path(stage.name)
            try:
                staged_path.write_bytes(payload)
                verify(staged_path, digest, files)
                # Atomic publication on NTFS. An existing destination is preserved.
                os.link(staged_path, args.archive)
            finally:
                staged_path.unlink(missing_ok=True)
        verify(args.archive, digest, files)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        parser.exit(2, f'{exc}\n')
    print(f'{commit}  {digest}  {args.archive}')


if __name__ == '__main__':
    main()
