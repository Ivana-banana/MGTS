"""Compare immutable runtime ZIPs on the same recorded inputs, one fresh process per phase."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def worker(archive, phase, data, config, out, resolution, initial):
    sys.path.insert(0, str(archive.resolve()))
    from mgts_nav.bag import load_config, scans
    from mgts_nav.benchmark import peak_rss_bytes
    from mgts_nav.delayed_start import search_prefix
    from mgts_nav.grid import read_map
    from mgts_nav.__main__ import main
    result = {}
    if phase == 'decode':
        result['scans'] = {name: sum(1 for _ in scans(data / name, load_config(config)))
                           for name in ('mapping', 'inspection')}
    elif phase == 'start':
        pixels, meta, _ = read_map(out / 'map/map.yaml')
        stream = scans(data / 'inspection', load_config(config))
        try:
            pose, _ = search_prefix(stream, 'delayed', pixels, meta, {})
            result['pose'] = pose.tolist()
        except ValueError as exc:
            result['refusal'] = str(exc)
    else:
        sys.argv = ['mgts_nav', 'map' if phase == 'map' else 'localize',
                    str(data / ('mapping' if phase == 'map' else 'inspection')), '--config', str(config),
                    '--out', str(out / phase)]
        if phase == 'map':
            sys.argv.extend(['--resolution', resolution])
        else:
            sys.argv.extend(['--map', str(out / 'map/map.yaml'), '--initial', *initial])
        main()
    result['peak_mib'] = peak_rss_bytes() / 2**20
    print(json.dumps(result, allow_nan=False))


def benchmark(args):
    from mgts_nav.benchmark import hardware
    from mgts_nav.metrics import evaluate, read_csv, wrap
    from mgts_nav.solve_manifest import bag_files, digest
    from tools.package_solution import build
    from importlib.metadata import version
    args.out.mkdir(parents=True, exist_ok=False)
    revision, payload = build(args.revision)
    archive = args.out / 'runtime.zip'
    archive.write_bytes(payload)
    needed = [args.data/'mapping', args.data/'inspection', args.config, args.truth]
    missing = [str(path) for path in needed if not path.exists()]
    if missing:
        unavailable = dict(revision=revision, status='unavailable', missing=missing)
        (args.out/'report.json').write_text(json.dumps(unavailable, indent=2), encoding='utf-8')
        print(json.dumps(unavailable))
        return
    inputs = [p for name in ('mapping', 'inspection') for p in bag_files((args.data / name).resolve())]
    truth = read_csv(args.truth)
    result = dict(revision=revision, archive_sha256=digest(archive), hardware=hardware(),
                  packages={n: version(n) for n in ('numpy', 'scipy', 'rosbags')},
                  resolution=args.resolution, inputs={str(p): digest(p) for p in [*inputs, args.config, args.truth]},
                  phases={})
    for phase in ('decode', 'map', 'start', 'inspection'):
        command = [sys.executable, '-I', str(Path(__file__).resolve()), '--worker', str(archive),
                   phase, str(args.data), str(args.config), str(args.out), str(args.resolution)]
        if phase == 'inspection':
            command.extend(map(str, truth[0][1:]))  # Separate known-start control on every map.
        started = perf_counter()
        output = subprocess.check_output(command, text=True)
        row = json.loads(output.splitlines()[-1])
        row.update(wall_s=perf_counter() - started, command=command)
        result['phases'][phase] = row
    pose = result['phases']['start'].get('pose')
    outcome = 'refused'
    if pose is not None:
        xy = math.dist(pose[:2], truth[0][1:3])
        yaw = abs(math.degrees(wrap(pose[2] - truth[0].yaw)))
        outcome = 'correct' if xy <= .25 and yaw <= 10 else 'wrong'
        result.update(start_xy_m=xy, start_yaw_deg=yaw)
    result.update(start_outcome=outcome, known_start=evaluate(read_csv(args.out/'inspection/trajectory.csv'), truth),
                  product_bytes=sum(p.stat().st_size for name in ('map', 'inspection')
                                    for p in (args.out/name).rglob('*') if p.is_file()))
    (args.out/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    if sys.argv[1:2] == ['--worker']:
        archive, phase, data, config, out, resolution, *initial = sys.argv[2:]
        worker(Path(archive), phase, Path(data), Path(config), Path(out), resolution, initial)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('revision')
        for name in ('data', 'config', 'truth', 'out'):
            parser.add_argument(name, type=Path)
        parser.add_argument('--resolution', type=float, default=.08)
        benchmark(parser.parse_args())
