"""Build or verify the MGTS runtime ZIP from an immutable Git revision."""
import argparse
import hashlib
import io
from pathlib import Path
import subprocess
import os
from tempfile import NamedTemporaryFile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    '__init__', '__main__', 'bag', 'benchmark', 'calibration', 'continuous', 'demo',
    'deskew', 'grid', 'initialization', 'leveling', 'loop', 'map_eval', 'metrics',
    'multiscan', 'observability', 'pose_graph', 'profile', 'quality', 'recovery', 'rehearse',
    'ros_export', 'slam', 'solve', 'start_confirmation', 'stress', 'vision', 'visual_fusion',
)
FILES = tuple(sorted((
    'README.md', 'requirements.txt', 'configs/example.json', 'configs/inspection.rviz',
    'docs/ambiguity.md', 'docs/corridor.md', 'docs/demo-results.md', 'docs/demo-results.png',
    'docs/intake.md', 'docs/public-data.md', 'docs/report.md', 'docs/runtime.md',
    'docs/sensors.md', 'docs/start-confirmation.md',
    *(f'mgts_nav/{name}.py' for name in MODULES),
)))


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def build(revision):
    commit = git('rev-parse', '--verify', '--end-of-options', revision + '^{commit}').decode().strip()
    entries = {}
    for record in git('ls-tree', '-r', '-z', commit, '--', *FILES).split(b'\0'):
        if record:
            header, name = record.split(b'\t', 1)
            mode, kind, oid = header.split()
            if mode not in (b'100644', b'100755') or kind != b'blob':
                raise ValueError(f'Expected a regular file: {name!r}')
            entries[name.decode()] = oid
    # Request immutable blobs directly so checkout filters cannot convert line endings.
    request = b''.join(entries[path] + b'\n' for path in FILES)
    blobs = io.BytesIO(subprocess.check_output(['git', 'cat-file', '--batch'], input=request, cwd=ROOT))
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w') as archive:
        for path in FILES:
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


def verify(path, expected_digest):
    with path.open('rb') as stream:
        actual = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            actual.update(block)
        if actual.hexdigest() != expected_digest:
            raise ValueError('SHA-256 differs from the selected revision')
        with zipfile.ZipFile(stream) as archive:
            if tuple(archive.namelist()) != FILES or archive.testzip() is not None:
                raise ValueError('ZIP membership or CRC mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--verify', action='store_true', help='Check an existing ZIP')
    args = parser.parse_args()
    try:
        commit, payload = build(args.revision)
        digest = hashlib.sha256(payload).hexdigest()
        if not args.verify:
            args.archive.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=args.archive.parent, delete=False) as stage:
                staged_path = Path(stage.name)
            try:
                staged_path.write_bytes(payload)
                verify(staged_path, digest)
                # Atomic publication on NTFS. An existing destination is preserved.
                os.link(staged_path, args.archive)
            finally:
                staged_path.unlink(missing_ok=True)
        verify(args.archive, digest)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        parser.exit(2, f'{exc}\n')
    print(f'{commit}  {digest}  {args.archive}')


if __name__ == '__main__':
    main()
