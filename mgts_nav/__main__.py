"""Offline navigation CLI."""
import argparse
from collections import Counter
import csv
from functools import partial
from itertools import chain, islice, tee
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .bag import inspect_bag, load_config, scans
from .grid import DEFAULT_RESOLUTION, Grid, preview, read_map
from .slam import VelocityPrior, checked_pose, read_initial_pose, relative, track, world


class MissingBaselineAnchor(ValueError):
    """The first scan has no synchronized kinematic position."""


def initial_pose(args):
    if args.command == 'map':
        return np.zeros(3)
    if args.initial is not None:
        return checked_pose(args.initial)
    if args.initial_file is not None:
        return read_initial_pose(args.initial_file)
    return None


def scan_rejections(estimates):
    rejected = {}
    for row in estimates:
        if row.rejection_reason is not None:
            rejected.setdefault(row.rejection_reason, []).append(row.t)
    return dict(counts={reason: len(times) for reason, times in rejected.items()}, timestamps_s=rejected)


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
    search_diagnostics = {}
    drift_recovery = None
    recovery = None
    if args.command == 'localize':
        from .safe_defaults import resolve_confirm_start, resolve_recover
        map_path = args.map
        pixels, meta, points = read_map(map_path)
        if getattr(args, 'recover_drift', False):
            from .recovery import DriftRecovery
            drift_recovery = DriftRecovery(pixels, meta)
        if resolve_recover(getattr(args, 'recover', None)):
            from .recovery import Recovery
            recovery = Recovery(pixels, meta)
        search = getattr(args, 'start_search', None)
        want_confirm = resolve_confirm_start(
            getattr(args, 'confirm_start', None),
            initial=None if automatic else start,
            start_search=search,
        )
        if search:
            from .delayed_start import search_prefix
            try:
                start, prefix = search_prefix(stream, search, pixels, meta, search_diagnostics)
            except ValueError as exc:
                report = dict(method=search, reason=str(exc), **search_diagnostics)
                raise ValueError(f'Start search refused: {json.dumps(report, allow_nan=False)}') from exc
            stream = chain(prefix, stream)
            start_method = search
        elif automatic:
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
                if want_confirm:
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
    start = checked_pose(start)
    if args.command == 'map':
        grid_factory = Grid
        if getattr(args, 'grid_dll', None):
            from .native_grid import NativeGrid, load_dll
            grid_factory = partial(NativeGrid, dll=load_dll(args.grid_dll))
        grid = grid_factory(args.resolution)
        endpoints = {}
        def capture_endpoints():
            for scan in stream:
                endpoints.setdefault('first', scan.xy)
                endpoints['last'] = scan.xy
                yield scan
        prior = VelocityPrior() if args.mapping_velocity_prior else None
        estimates = list(track(capture_endpoints(), start, grid=grid, velocity_prior=prior))
        if not estimates:
            raise ValueError(f'No lidar messages on {cfg["lidar"]} (scans=0)')
        rejections = scan_rejections(estimates)
        map_counts = Counter(row.status for row in estimates)
        detail = (f'scans={len(estimates)}, status={json.dumps(map_counts)}, '
                  f'rejections={json.dumps(rejections["counts"])}')
        if not map_counts['origin']:
            raise ValueError(f'No usable scan initialized the map: {detail}')
        if getattr(args, 'internal_loops', False):
            from .interior_loop import search
            from .pose_graph import adjust_internal
            rows = (dict(t=r.t, x=r.pose[0], y=r.pose[1], yaw=r.pose[2], status=r.status) for r in estimates)
            report = search(scans(args.bag, cfg), rows)
            poses, selected = adjust_internal([r.pose for r in estimates], report['candidates'])
            report['selected'] = selected
            if selected:
                grid = grid_factory(args.resolution)
                for scan, row, pose in zip(scans(args.bag, cfg), estimates, poses, strict=True):
                    row.pose = pose
                    if row.status != 'prediction_only':
                        grid.observe(world(scan.origin, pose), world(scan.xy, pose))
            (output / 'internal-loops.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if args.close_loop:
            from .loop import diagnose
            from .pose_graph import adjust
            if len(estimates) < 3:
                raise ValueError(f'Too few estimates to evaluate endpoint loop: {detail}')
            poses = np.array([r.pose for r in estimates])
            loop = diagnose(endpoints['first'], endpoints['last'], poses)
            if loop['loop_supported']:
                poses = adjust(poses, loop['measured_first_from_last'])
                grid = grid_factory(args.resolution)
                for scan, row, pose in zip(scans(args.bag, cfg), estimates, poses, strict=True):
                    row.pose = pose
                    if row.status != 'prediction_only':
                        grid.observe(world(scan.origin, pose), world(scan.xy, pose))
            (output / 'loop.json').write_text(json.dumps(loop, indent=2), encoding='utf-8')
        try:
            grid.save(output)
        except ValueError as exc:
            raise ValueError(f'{exc}: {detail}') from exc
        map_path = output / 'map.yaml'
    elif args.command == 'localize':
        estimates = list(track(stream, start, map_points=points, recovery=recovery, drift_recovery=drift_recovery))
        rejections = scan_rejections(estimates)
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
    if args.command == 'localize' and not counts['matched'] and not counts['uncertain']:
        raise ValueError('No scan matched the map. Check initial pose, frames and height band. '
                         f'Rejections: {json.dumps(rejections["counts"])}')
    with (output / 'trajectory.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'x', 'y', 'yaw', 'status', 'inlier_fraction'])
        for r in estimates:
            writer.writerow([r.t, *r.pose, r.status, r.inliers])
    preview(map_path, output / 'route.png', estimates)
    summary = dict(scans=len(estimates), status=counts, final_pose=estimates[-1].pose.tolist(),
                   final_status=estimates[-1].status, initial_pose=start.tolist(),
                   initialization=start_method, scan_rejections=rejections)
    if args.command == 'map' and prior is not None:
        summary['mapping_velocity_prior_steps'] = prior.applied
    if search_diagnostics:
        summary['start_search'] = search_diagnostics
    if args.command == 'localize':
        from .quality import inspection_quality
        summary['quality'] = inspection_quality(estimates)
    if visual_stats:
        summary['visual_yaw'] = visual_stats
    if drift_recovery is not None:
        summary['drift_recovery'] = drift_recovery.stats
    if recovery is not None:
        summary['recovery'] = recovery.stats
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(prog='python -m mgts_nav', description=__doc__,
                                     epilog='Other modules: mgts_nav.solve, mgts_nav.profile, '
                                            'mgts_nav.metrics, mgts_nav.rehearse.')
    sub = parser.add_subparsers(dest='command', required=True)
    survey = sub.add_parser('inspect', help='List topics, frames, fields, timing and gaps')
    bag_help = 'Recording directory containing metadata.yaml or a supported recording file'
    survey.add_argument('bag', type=Path, help=bag_help)
    survey.add_argument('--inventory', action='store_true', help='Read metadata only; skip message payloads')
    for name, description in (('map', 'Build an occupancy map'),
                              ('localize', 'Localize the inspection recording'),
                              ('baseline', 'Dead-reckon from odometry')):
        p = sub.add_parser(name, help=description, description=description)
        p.add_argument('bag', type=Path, help=bag_help)
        p.add_argument('--config', required=True, type=Path)
        p.add_argument('--out', required=True, type=Path, help='A new output directory')
        if name != 'baseline':
            p.add_argument('--visual-pairs', type=Path, help='Optional calibrated feature pairs in JSONL')
        if name == 'map':
            p.add_argument('--mapping-velocity-prior', action='store_true',
                           help='Predict mapping translation from two matches when odometry is missing')
            p.add_argument('--grid-dll', type=Path, help='Use a locally built MGTS native occupancy-grid DLL')
            p.add_argument('--resolution', default=DEFAULT_RESOLUTION, type=float, help='Grid cell size in metres')
            loops = p.add_mutually_exclusive_group()
            loops.add_argument('--internal-loops', action='store_true', help='Experimentally correct supported internal mapping revisits')
            loops.add_argument('--close-loop', action='store_true', help='Optimize poses if endpoint revisit is supported')
        else:
            group = p.add_mutually_exclusive_group(required=name == 'baseline')
            group.add_argument('--initial', type=float, nargs=3, metavar=('X', 'Y', 'YAW'),
                               help='Map-frame X and Y in metres, YAW in radians')
            group.add_argument('--initial-file', type=Path, help='JSON array [x,y,yaw] in map')
        if name == 'localize':
            from .delayed_start import START_SEARCHES
            group.add_argument('--start-search', choices=START_SEARCHES, help='Choose a global pose search')
            p.add_argument('--map', type=Path, required=True, help='map.yaml from mapping')
            p.add_argument('--recover-drift', action='store_true', help='Experimentally recheck a matched pose after a drift warning')
            confirm = p.add_mutually_exclusive_group()
            confirm.add_argument(
                '--confirm-start', action='store_true', dest='confirm_start',
                help='Verify an accepted single-scan start using later scans (default for automatic start)')
            confirm.add_argument(
                '--no-confirm-start', action='store_false', dest='confirm_start',
                help='Accept a single-scan start without later confirmation')
            recover = p.add_mutually_exclusive_group()
            recover.add_argument(
                '--recover', action='store_true', dest='recover',
                help='Try confirmed global recovery after loss (default)')
            recover.add_argument(
                '--no-recover', action='store_false', dest='recover',
                help='Disable global recovery after localization loss')
            p.set_defaults(confirm_start=None, recover=None)
    args = parser.parse_args()
    try:
        if args.command == 'inspect':
            print(json.dumps(inspect_bag(args.bag, args.inventory), indent=2))
            return
        if args.out.exists():
            parser.exit(2, f'Output already exists: {args.out}. Choose a new directory.\n')
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
    try:
        main()
    except KeyboardInterrupt:
        import sys
        print('Interrupted', file=sys.stderr)
        sys.exit(130)
