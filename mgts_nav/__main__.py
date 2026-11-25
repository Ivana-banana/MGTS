"""Offline navigation CLI."""
import argparse
from collections import Counter
import csv
from itertools import chain, islice, tee
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .bag import inspect_bag, load_config, scans
from .grid import Grid, preview, read_map
from .slam import relative, track, world


class MissingBaselineAnchor(ValueError):
    """The first scan has no synchronized kinematic position."""


def initial_pose(args):
    if args.command == 'map':
        return np.zeros(3)
    if args.initial is not None:
        return np.array(args.initial)
    if args.initial_file is not None:
        return np.array(json.loads(args.initial_file.read_text(encoding='utf-8-sig')), dtype=float)
    return None


def run(args, output):
    cfg = load_config(args.config)
    stream = scans(args.bag, cfg)
    visual_stats = {}
    if getattr(args, 'visual_pairs', None):
        from .visual_fusion import apply_visual
        stream = apply_visual(stream, args.visual_pairs, visual_stats)
    start = initial_pose(args)
    automatic = start is None
    start_method = 'map_origin' if args.command == 'map' else 'provided'
    recovery = None
    if args.command == 'localize':
        map_path = args.map
        pixels, meta, points = read_map(map_path)
        if args.recover:
            from .recovery import Recovery
            recovery = Recovery(pixels, meta)
        if automatic:
            from .initialization import estimate_initial
            first = next(stream, None)
            if first is None:
                raise ValueError('No lidar scans in recording')
            prefix = [first]
            try:
                start = estimate_initial(first.xy, pixels, meta)
            except ValueError as single_error:
                from .multiscan import MAX_SCANS, estimate_multiscan
                prefix.extend(islice(stream, MAX_SCANS - 1))
                try:
                    start = estimate_multiscan(prefix, pixels, meta)
                except ValueError as multi_error:
                    raise ValueError(f'Provide --initial or check calibration. Single scan: {single_error}. '
                                     f'Multiple scans: {multi_error}') from multi_error
                start_method = 'multiple_scans_to_map'
            else:
                start_method = 'scan_to_map'
                if args.confirm_start:
                    from .start_confirmation import confirm_start
                    following, stream = tee(stream)
                    try:
                        start = confirm_start(start, first, following, pixels, meta)
                    except ValueError as error:
                        raise ValueError(f'Could not confirm automatic start: {error}. '
                                         'Provide --initial or check calibration.') from error
                    finally:
                        del following
                    start_method = 'confirmed_scan_to_map'
            stream = chain(prefix, stream)
    if start.shape != (3,) or not np.isfinite(start).all():
        raise ValueError('initial pose must contain finite x, y, yaw')
    if args.command == 'map':
        grid = Grid(args.resolution)
        endpoints = {}
        def capture_endpoints():
            for scan in stream:
                endpoints.setdefault('first', scan.xy)
                endpoints['last'] = scan.xy
                yield scan
        estimates = list(track(capture_endpoints(), start, grid=grid))
        if args.close_loop:
            from .loop import diagnose
            from .pose_graph import adjust
            if len(estimates) < 3:
                raise ValueError('Too few estimates to evaluate endpoint loop')
            poses = np.array([r.pose for r in estimates])
            loop = diagnose(endpoints['first'], endpoints['last'], poses)
            if loop['loop_supported']:
                poses = adjust(poses, loop['measured_first_from_last'])
                grid = Grid(args.resolution)
                for scan, row, pose in zip(scans(args.bag, cfg), estimates, poses, strict=True):
                    row.pose = pose
                    if row.status != 'prediction_only':
                        grid.observe(world(scan.origin, pose), world(scan.xy, pose))
            (output / 'loop.json').write_text(json.dumps(loop, indent=2), encoding='utf-8')
        try:
            grid.save(output)
        except ValueError as error:
            reasons = Counter(r.rejection_reason for r in estimates if r.rejection_reason is not None)
            raise ValueError(f'{error}. Scans: {len(estimates)}, used: {len(estimates) - sum(reasons.values())}. '
                             f'Rejections: {json.dumps(dict(reasons))}. Check frames, height band, '
                             'point_time unit and calibration (python -m mgts_nav.calibration --floor).') from error
        map_path = output / 'map.yaml'
    elif args.command == 'localize':
        estimates = list(track(stream, start, map_points=points, recovery=recovery))
    else:
        first = None
        with (output / 'trajectory.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['t', 'x', 'y', 'yaw'])
            for scan in stream:
                if first is None:
                    if scan.odom is None:
                        raise MissingBaselineAnchor('No odometry at first scan to anchor baseline')
                    first = scan.odom
                if scan.odom is not None:
                    delta = relative(first, scan.odom)
                    writer.writerow([scan.t, *world(delta[:2], start), start[2] + delta[2]])
        if first is None:
            raise ValueError('No lidar scans in recording')
        return
    if not estimates:
        raise ValueError('No scans in recording')
    counts = Counter(r.status for r in estimates)
    if args.command == 'localize':
        rejected = {}
        for row in estimates:
            if row.rejection_reason is not None:
                rejected.setdefault(row.rejection_reason, []).append(row.t)
        rejection_counts = {reason: len(times) for reason, times in rejected.items()}
        if not counts['matched']:
            raise ValueError('No scan matched the map. Check initial pose, frames and height band. '
                             f'Rejections: {json.dumps(rejection_counts)}')
    with (output / 'trajectory.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'x', 'y', 'yaw', 'status', 'inlier_fraction'])
        for r in estimates:
            writer.writerow([r.t, *r.pose, r.status, r.inliers])
    preview(map_path, output / 'route.png', estimates)
    summary = dict(scans=len(estimates), status=counts, final_pose=estimates[-1].pose.tolist(),
                   final_status=estimates[-1].status, initial_pose=start.tolist(),
                   initialization=start_method)
    if args.command == 'localize':
        from .quality import inspection_quality
        summary['quality'] = inspection_quality(estimates)
        summary['scan_rejections'] = dict(counts=rejection_counts, timestamps_s=rejected)
    if visual_stats:
        summary['visual_yaw'] = visual_stats
    if recovery is not None:
        summary['recovery'] = recovery.stats
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    survey = sub.add_parser('inspect', help='List topics, frames, fields, timing and gaps')
    survey.add_argument('bag', type=Path)
    survey.add_argument('--inventory', action='store_true', help='Read metadata only; skip message payloads')
    for name in ('map', 'localize', 'baseline'):
        p = sub.add_parser(name)
        p.add_argument('bag', type=Path)
        p.add_argument('--config', required=True, type=Path)
        p.add_argument('--out', required=True, type=Path, help='A new output directory')
        if name != 'baseline':
            p.add_argument('--visual-pairs', type=Path, help='Optional calibrated feature pairs in JSONL')
        if name == 'map':
            p.add_argument('--resolution', default=0.08, type=float, help='Grid cell size in metres')
            p.add_argument('--close-loop', action='store_true', help='Optimize poses if endpoint revisit is supported')
        else:
            group = p.add_mutually_exclusive_group(required=name == 'baseline')
            group.add_argument('--initial', type=float, nargs=3, metavar=('X', 'Y', 'YAW'))
            group.add_argument('--initial-file', type=Path, help='JSON array [x,y,yaw] in map')
        if name == 'localize':
            group.add_argument('--confirm-start', action='store_true',
                               help='Verify an accepted single-scan start using later scans and odometry')
            p.add_argument('--map', type=Path, required=True, help='map.yaml from mapping')
            p.add_argument('--recover', action='store_true', help='Try confirmed global recovery after loss')
    args = parser.parse_args()
    try:
        if args.command == 'inspect':
            print(json.dumps(inspect_bag(args.bag, args.inventory), indent=2))
            return
        if args.out.exists():
            parser.error(f'Output already exists: {args.out}. Choose a new directory.')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.out.parent) as tmp:
            stage = Path(tmp) / 'result'
            stage.mkdir()
            run(args, stage)
            stage.rename(args.out)
    except MissingBaselineAnchor as error:
        parser.exit(3, f'{error}\n')
    except (OSError, KeyError, ValueError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
