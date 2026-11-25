"""Process an official mapping/inspection bag pair on CPU, without ground truth."""
import argparse
import json
from math import isfinite
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from .bag import check_recording
from .delayed_start import START_SEARCHES
from .ros_export import export
from .slam import checked_pose, read_initial_pose
from .zones import load_zones, read_visits, write_visits


def module_command(module, *args):
    python = [sys.executable, '-X', f'utf8={sys.flags.utf8_mode}']
    archive = getattr(__loader__, 'archive', None)
    if archive:
        # A fresh interpreter does not inherit the parent's zipimport search path.
        bootstrap = ("import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); "
                     "runpy.run_module(sys.argv.pop(1), run_name='__main__', alter_sys=True)")
        return [*python, '-I', '-c', bootstrap, str(Path(archive).resolve()), module, *map(str, args)]
    return [*python, '-m', module, *map(str, args)]


def solve(mapping, inspection, profile, output, inspection_profile=None, initial=None, close_loop=False,
          mapping_visual_pairs=None, inspection_visual_pairs=None, recover=False, confirm_start=False, grid_dll=None,
          start_search=None, zones=None, zone_max_gap=None):
    if (zones is None) != (zone_max_gap is None):
        raise ValueError('Supply --zones and --zone-max-gap together')
    if zone_max_gap is not None and (not isfinite(zone_max_gap) or zone_max_gap <= 0):
        raise ValueError('--zone-max-gap must be positive and finite')
    if start_search is not None and (start_search not in START_SEARCHES or initial is not None or confirm_start):
        raise ValueError('--start-search requires a supported search and conflicts with --initial and --confirm-start')
    if initial is not None and confirm_start:
        raise ValueError('--confirm-start requires automatic initialization. Omit --initial.')
    if initial is not None:
        initial = checked_pose(initial)
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
        if mapping_visual_pairs is not None:
            mapping_visual_pairs = shutil.copyfile(mapping_visual_pairs, profiles / 'mapping-pairs.jsonl')
        if inspection_visual_pairs is not None:
            inspection_visual_pairs = shutil.copyfile(inspection_visual_pairs, profiles / 'inspection-pairs.jsonl')
        if zones is not None:
            shutil.copyfile(zones, profiles / 'zones.json')
            polygons = load_zones(profiles / 'zones.json')
        check_recording(mapping, profiles / 'mapping.json')
        cfg = check_recording(inspection, profiles / 'inspection.json')
        def invoke(*args):
            subprocess.run(module_command('mgts_nav', *args), check=True)
        map_flags = ['--close-loop'] if close_loop else []
        if grid_dll is not None:
            map_flags.extend(['--grid-dll', grid_dll])
        if mapping_visual_pairs:
            map_flags.extend(['--visual-pairs', mapping_visual_pairs])
        invoke('map', mapping, '--config', profiles / 'mapping.json', '--out', map_dir, *map_flags)
        options = ['--initial', *initial] if initial is not None else []
        if start_search is not None:
            options.extend(['--start-search', start_search])
        if inspection_visual_pairs:
            options.extend(['--visual-pairs', inspection_visual_pairs])
        if confirm_start:
            options.append('--confirm-start')
        if recover:
            options.append('--recover')
        invoke('localize', inspection, '--config', profiles / 'inspection.json',
               '--map', map_dir / 'map.yaml', '--out', result, *options)
        summary = json.loads((result / 'summary.json').read_text(encoding='utf-8'))
        # Both methods get the same start, including when it was estimated by scan matching.
        route = result / 'trajectory.csv'
        if zones is not None:
            entries = read_visits(route, polygons, zone_max_gap)
            write_visits(result / 'zone-visits.csv', entries)
            summary['zones'] = dict(max_gap_s=zone_max_gap, visit_intervals=len(entries))
            (result / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        if cfg['odom'] is not None:
            baseline = subprocess.run(module_command('mgts_nav', 'baseline', str(inspection),
                                       '--config', str(profiles / 'inspection.json'), '--initial',
                                       *map(str, summary['initial_pose']), '--out', str(stage / 'baseline')),
                                      capture_output=True, text=True)
            if baseline.returncode:
                # The two main products remain usable when the comparison lacks an anchor.
                if baseline.returncode != 3:
                    raise ValueError(baseline.stderr.strip() or 'Baseline failed')
                (stage / 'baseline-status.txt').write_text(baseline.stderr.strip(), encoding='utf-8')
            dense = result / 'continuous.csv'
            continuous = subprocess.run(module_command('mgts_nav.continuous', inspection,
                                         route, dense, '--config', profiles / 'inspection.json'),
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
    parser.add_argument('--zones', type=Path, help='Map-frame inspection zone polygons')
    parser.add_argument('--zone-max-gap', type=float, metavar='SECONDS', help='Largest gap allowed inside a zone visit')
    start = parser.add_mutually_exclusive_group()
    start.add_argument('--start-search', choices=START_SEARCHES)
    start.add_argument('--confirm-start', action='store_true',
                       help='Verify an accepted single-scan start using later scans and odometry')
    start.add_argument('--initial', type=float, nargs=3, metavar=('X', 'Y', 'YAW'))
    start.add_argument('--initial-file', type=Path, help='JSON array [x,y,yaw] in map')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--close-loop', action='store_true')
    parser.add_argument('--grid-dll', type=Path, help='Use a locally built MGTS occupancy-grid DLL for mapping')
    parser.add_argument('--mapping-visual-pairs', type=Path)
    parser.add_argument('--inspection-visual-pairs', type=Path)
    parser.add_argument('--recover', action='store_true')
    args = parser.parse_args()
    try:
        if args.initial_file is not None:
            args.initial = read_initial_pose(args.initial_file)
        solve(args.mapping, args.inspection, args.config, args.out, args.inspection_config,
              args.initial, args.close_loop, args.mapping_visual_pairs, args.inspection_visual_pairs,
              args.recover, args.confirm_start, args.grid_dll, args.start_search, args.zones, args.zone_max_gap)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f'{exc}\n')
