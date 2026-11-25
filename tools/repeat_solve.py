"""Repeat a completed solve with its archived profiles and recorded options."""
import argparse
import json
from pathlib import Path
import subprocess

from mgts_nav.solve import solve
from mgts_nav.solve_manifest import digest


FLAGS = ('close_loop', 'mapping_velocity_prior', 'recover', 'confirm_start',
         'mapping_visual_pairs', 'inspection_visual_pairs', 'zones')


def repeat(previous, mapping, inspection, output, dll=None):
    settings = json.loads((previous / 'solve-options.json').read_bytes())
    required = {*FLAGS, 'resolution', 'initial', 'start_search', 'grid_backend',
                'grid_dll_sha256', 'zone_max_gap'}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError('Incomplete or unsupported solve-options.json')
    if any(type(settings[k]) is not bool for k in FLAGS):
        raise ValueError('Expected boolean flags in solve-options.json')
    backend = settings.pop('grid_backend')
    checksum = settings.pop('grid_dll_sha256')
    if backend == 'cpp':
        if dll is None or digest(dll) != checksum:
            raise ValueError('Supply --grid-dll with the recorded SHA-256')
    elif backend != 'python' or dll is not None or checksum is not None:
        raise ValueError('Inconsistent grid backend or DLL settings')
    profiles = previous / 'profiles'
    for key, name in [('mapping_visual_pairs', 'mapping-pairs.jsonl'),
                      ('inspection_visual_pairs', 'inspection-pairs.jsonl'), ('zones', 'zones.json')]:
        settings[key] = profiles / name if settings[key] else None
    return solve(mapping, inspection, profiles / 'mapping.json', output,
                 inspection_profile=profiles / 'inspection.json', grid_dll=dll, **settings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('previous', 'mapping', 'inspection', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--grid-dll', type=Path)
    args = parser.parse_args()
    try:
        repeat(args.previous, args.mapping, args.inspection, args.output, args.grid_dll)
    except (OSError, ValueError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(2, f'{error}\n')
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted\n')
