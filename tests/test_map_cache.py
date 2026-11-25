import pytest

from mgts_nav.map_cache import identity, use_or_build


def test_cache_dependencies_and_integrity(tmp_path):
    bag = tmp_path / 'bag.mcap'
    profile = tmp_path / 'profile.json'
    pairs = tmp_path / 'motion.jsonl'
    for path in (bag, profile, pairs):
        path.write_text('original')
    settings = dict(resolution=.08, mapping_velocity_prior=False, grid_backend='python',
                    grid_dll_sha256=None, close_loop=False)

    def key():
        return identity(bag, profile, pairs, settings)

    cache = tmp_path / 'saved'
    target = tmp_path / 'first'

    def build(directory):
        directory.mkdir()
        for name in ('map.yaml', 'trajectory.csv', 'summary.json', 'route.png', 'map.pgm'):
            (directory / name).write_bytes(b'test payload')

    use_or_build(cache, target, key(), build)
    def unexpected_build(_):
        pytest.fail('A completed map must be reused')

    for path in (bag, profile, pairs):
        path.write_text('changed')
        with pytest.raises(ValueError, match='inputs or algorithm'):
            use_or_build(cache, tmp_path / 'retry', key(), unexpected_build)
        path.write_text('original')
    for name, value in [('resolution', .12), ('mapping_velocity_prior', True)]:
        old = settings[name]
        settings[name] = value
        with pytest.raises(ValueError, match='inputs or algorithm'):
            use_or_build(cache, tmp_path / 'retry', key(), unexpected_build)
        settings[name] = old
    use_or_build(cache, tmp_path / 'retry', key(), unexpected_build)
    (cache / 'map/map.pgm').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        use_or_build(cache, tmp_path / 'damaged', key(), unexpected_build)


def test_interrupted_mapping_is_not_published(tmp_path):
    def interrupt(directory):
        directory.mkdir()
        (directory / 'map.yaml').touch()
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        use_or_build(tmp_path / 'cache', tmp_path / 'map', {}, interrupt)
    assert not list(tmp_path.iterdir())
