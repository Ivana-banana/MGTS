"""Process an official mapping/inspection bag pair on CPU, without ground truth."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .bag import check_recording, load_config
from .metrics import read_csv
from .ros_export import export

# Both route measures must exceed the mapping route by this factor to call the order suspicious.
SWAP_RATIO = 1.25


def route_extent(path):
    """Duration, path length and convex-hull area of a trajectory CSV, all in the map frame."""
    rows = read_csv(path)
    xy = np.array([[r.x, r.y] for r in rows])
    try:
        area = float(ConvexHull(xy).volume) if len(xy) >= 3 else 0.0
    except QhullError:
        area = 0.0
    return dict(duration_s=rows[-1].t - rows[0].t,
                path_m=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()), hull_area_m2=area)


def recording_order(mapping_route, inspection_route):
    """Flag a likely swap of the two recordings. Heuristic only: it never stops the run."""
    mapping, inspection = route_extent(mapping_route), route_extent(inspection_route)
    suspected = all(inspection[key] > SWAP_RATIO * mapping[key] for key in ('path_m', 'hull_area_m2'))
    report = dict(mapping=mapping, inspection=inspection, ratio=SWAP_RATIO, suspected_swap=suspected)
    if suspected:
        report['warning'] = (
            f'The inspection route is longer ({inspection["path_m"]:.1f} m vs {mapping["path_m"]:.1f} m) '
            f'and covers more area ({inspection["hull_area_m2"]:.1f} m2 vs {mapping["hull_area_m2"]:.1f} m2) '
            'than the mapping route. Check that the recordings are not swapped: '
            'the first positional argument is the mapping recording, the second is the inspection.')
    return report


def solve(mapping, inspection, profile, output, inspection_profile=None, initial=None, close_loop=False,
          mapping_visual_pairs=None, inspection_visual_pairs=None, recover=False, confirm_start=False):
    if initial is not None and confirm_start:
        raise ValueError('--confirm-start requires automatic initialization. Omit --initial.')
    # Validate the user's files first, so that errors name them and not the staged copies.
    load_config(profile)
    if inspection_profile is not None:
        load_config(inspection_profile)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / 'solution'
        stage.mkdir()
        map_dir = stage / 'map'
        result = stage / 'inspection'
        profiles = stage / 'profiles'
        profiles.mkdir()
        shutil.copyfile(profile, profiles / 'mapping.json')
        shutil.copyfile(inspection_profile or profile, profiles / 'inspection.json')
        check_recording(mapping, profiles / 'mapping.json')
        cfg = check_recording(inspection, profiles / 'inspection.json')
        def invoke(*args):
            subprocess.run([sys.executable, '-m', 'mgts_nav', *map(str, args)], check=True)
        map_flags = ['--close-loop'] if close_loop else []
        if mapping_visual_pairs:
            map_flags.extend(['--visual-pairs', mapping_visual_pairs])
        invoke('map', mapping, '--config', profiles / 'mapping.json', '--out', map_dir, *map_flags)
        options = ['--initial', *initial] if initial is not None else []
        if inspection_visual_pairs:
            options.extend(['--visual-pairs', inspection_visual_pairs])
        if confirm_start:
            options.append('--confirm-start')
        if recover:
            options.append('--recover')
        invoke('localize', inspection, '--config', profiles / 'inspection.json',
               '--map', map_dir / 'map.yaml', '--out', result, *options)
        summary = json.loads((result / 'summary.json').read_text(encoding='utf-8'))
        order = recording_order(map_dir / 'trajectory.csv', result / 'trajectory.csv')
        summary['recording_order'] = order
        (result / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        if order['suspected_swap']:
            print(f'WARNING: {order["warning"]}', file=sys.stderr)
        # Both methods get the same start, including when it was estimated by scan matching.
        route = result / 'trajectory.csv'
        if cfg['odom'] is not None:
            baseline = subprocess.run([sys.executable, '-m', 'mgts_nav', 'baseline', str(inspection),
                                       '--config', str(profiles / 'inspection.json'), '--initial',
                                       *map(str, summary['initial_pose']), '--out', str(stage / 'baseline')],
                                      capture_output=True, text=True)
            if baseline.returncode:
                # The two main products remain usable when the comparison lacks an anchor.
                if baseline.returncode != 3:
                    raise ValueError(baseline.stderr.strip() or 'Baseline failed')
                (stage / 'baseline-status.txt').write_text(baseline.stderr.strip(), encoding='utf-8')
            dense = result / 'continuous.csv'
            continuous = subprocess.run([sys.executable, '-m', 'mgts_nav.continuous', str(inspection),
                                         str(route), str(dense), '--config', str(profiles / 'inspection.json')],
                                        capture_output=True, text=True)
            if continuous.returncode:
                raise ValueError(continuous.stderr.strip() or 'Continuous trajectory failed')
            (result / 'continuous-summary.json').write_text(continuous.stdout, encoding='utf-8')
            route = dense
        export(map_dir / 'map.yaml', route, stage / 'rviz', 'map')
        stage.rename(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mapping', type=Path)
    parser.add_argument('inspection', type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--inspection-config', type=Path, help='Override mapping sensor profile')
    start = parser.add_mutually_exclusive_group()
    start.add_argument('--confirm-start', action='store_true',
                       help='Verify an accepted single-scan start using later scans and odometry')
    start.add_argument('--initial', type=float, nargs=3, metavar=('X', 'Y', 'YAW'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--close-loop', action='store_true')
    parser.add_argument('--mapping-visual-pairs', type=Path)
    parser.add_argument('--inspection-visual-pairs', type=Path)
    parser.add_argument('--recover', action='store_true')
    args = parser.parse_args()
    try:
        solve(args.mapping, args.inspection, args.config, args.out, args.inspection_config,
              args.initial, args.close_loop, args.mapping_visual_pairs, args.inspection_visual_pairs,
              args.recover, args.confirm_start)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f'{exc}\n')
