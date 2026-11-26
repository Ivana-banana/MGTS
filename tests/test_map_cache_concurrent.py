from concurrent.futures import ThreadPoolExecutor
import errno
import os
from pathlib import Path
from threading import Barrier

from mgts_nav.map_cache import products, use_or_build


def test_concurrent_cache_publication(tmp_path, monkeypatch):
    cache = tmp_path / 'cache'
    barrier = Barrier(2)
    rename = Path.rename

    if os.name == 'nt':
        def linux_collision(source, destination):
            try:
                return rename(source, destination)
            except FileExistsError as exc:
                # Exercise Linux's nonempty-directory errno on Windows too.
                raise OSError(errno.ENOTEMPTY, 'Directory not empty') from exc

        monkeypatch.setattr(Path, 'rename', linux_collision)

    def build(directory):
        directory.mkdir()
        for name in ('map.yaml', 'map.pgm', 'trajectory.csv', 'summary.json', 'route.png'):
            (directory / name).write_bytes(b'complete map')
        barrier.wait(timeout=10)

    targets = [tmp_path / 'first', tmp_path / 'second']
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(use_or_build, cache, target, {'same': 'key'}, build)
                   for target in targets]
        for future in futures:
            future.result(timeout=15)

    assert products(targets[0]) == products(targets[1]) == products(cache / 'map')
    assert len(products(targets[0])) == 5
