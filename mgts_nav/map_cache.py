"""Publish a complete mapping stage atomically and verify it before inspection retries."""
import hashlib
from importlib.metadata import version
from importlib.resources import files
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

from .solve_manifest import bag_files, digest


def identity(mapping, profile, pairs, settings):
    mapping = Path(mapping).resolve()
    sources = sorted((p for p in files('mgts_nav').iterdir() if p.name.endswith('.py')),
                     key=lambda p: p.name)
    return dict(
        format=1, python=list(sys.version_info[:3]),
        packages={name: version(name) for name in ('numpy', 'scipy', 'rosbags', 'PyYAML', 'Pillow')},
        algorithm={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        recording={p.relative_to(mapping).as_posix() if mapping.is_dir() else 'recording': digest(p)
                   for p in bag_files(mapping)},
        profile=digest(profile), visual_pairs=digest(pairs) if pairs else None,
        settings={k: settings[k] for k in ('resolution', 'mapping_velocity_prior', 'grid_backend',
                                            'grid_dll_sha256', 'close_loop')})


def products(directory):
    hashes = {p.relative_to(directory).as_posix(): digest(p) for p in directory.rglob('*') if p.is_file()}
    if not {'map.yaml', 'map.pgm', 'trajectory.csv', 'summary.json', 'route.png'} <= hashes.keys():
        raise ValueError(f'Incomplete cached map: {directory}')
    return hashes


def use_or_build(cache, target, key, build):
    cache = Path(cache)
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=cache.parent) as tmp:
            stage = Path(tmp) / 'candidate'
            stage.mkdir()
            build(stage / 'map')
            manifest = dict(inputs=key, products=products(stage / 'map'))
            (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            try:
                stage.rename(cache)
            except FileExistsError:
                pass  # Another complete writer won publication. Verify its copy below.
    manifest = json.loads((cache / 'manifest.json').read_bytes())
    if manifest.get('inputs') != key:
        raise ValueError(f'Cached mapping inputs or algorithm differ: {cache}. Choose a new --map-cache.')
    shutil.copytree(cache / 'map', target)
    if products(target) != manifest.get('products'):
        raise ValueError(f'Cached map failed checksum verification: {cache}')
