"""Replay identical scan endpoints through Grid.observe and the experimental C++ DLL."""
import argparse
import ctypes
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from mgts_nav.bag import load_config, scans
from mgts_nav.grid import Grid
from mgts_nav.slam import track


def load_dll(path):
    dll = ctypes.CDLL(str(path.resolve()))
    handle, size = ctypes.c_void_p, ctypes.c_size_t
    array = np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS')
    for name, result, args in (
        ('grid_create', handle, []), ('grid_destroy', None, [handle]),
        ('grid_observe', ctypes.c_int, [handle, array, array, size]),
        ('grid_size', size, [handle]), ('grid_export', ctypes.c_int, [handle, array, size]),
    ):
        function = getattr(dll, name)
        function.restype, function.argtypes = result, args
    return dll


def cells32(points, resolution):
    cells = np.floor(points / resolution)
    if not ((cells >= -2**31) & (cells < 2**31)).all():
        raise ValueError('Cell coordinate outside int32 range')
    return np.ascontiguousarray(cells, dtype=np.int32)


def native_evidence(inputs, resolution, dll):
    handle = dll.grid_create()
    if not handle:
        raise MemoryError('Creating native grid')
    try:
        for origin, points in inputs:
            near = points[np.linalg.norm(points - origin, axis=1) <= 35]
            cells = np.unique(cells32(near, resolution), axis=0)
            if dll.grid_observe(handle, cells32(origin, resolution), cells, len(cells)):
                raise RuntimeError('Updating native grid')
        rows = np.empty((dll.grid_size(handle), 3), dtype=np.int32)
        if dll.grid_export(handle, rows, len(rows)):
            raise RuntimeError('Exporting native grid')
        return {(x, y): value for x, y, value in rows.tolist()}
    finally:
        dll.grid_destroy(handle)


def python_evidence(inputs, resolution):
    grid = Grid(resolution)
    for origin, points in inputs:
        grid.observe(origin, points)
    return grid.evidence


def verify_edge_cases(dll):
    # Odd origin and signed half-cell ties exercise rounding relative to the ray start.
    # A shorter collinear ray ends inside a longer one. Repeat to reach both vote limits.
    origin = np.array([-.75, 1.25])
    ends = origin + np.array([(4, 2), (-4, -2), (-2, 4), (2, -4), (2, 1),
                              (0, 0), (4, 2), (36, 0), (0, 35)])
    inputs = [(origin, ends[:0]), (origin, ends)] * 16
    if native_evidence(inputs, 1., dll) != python_evidence(inputs, 1.):
        raise ValueError('Evidence mismatch in rounding, range or saturation controls')


class RegisteredScans(list):
    def observe(self, origin, endpoints):
        self.append((origin, endpoints))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--dll', required=True, type=Path)
    parser.add_argument('--resolution', type=float, default=.08)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('Expected at least one repeat')
    Grid(args.resolution)  # Apply the production resolution bounds before collecting scans.
    dll = load_dll(args.dll)
    verify_edge_cases(dll)
    inputs = RegisteredScans()
    estimates = list(track(scans(args.bag, load_config(args.config)), np.zeros(3), grid=inputs))
    if not inputs:
        parser.error('No accepted scans')
    times = {'python': [], 'cpp': []}
    for repeat in range(args.repeats):
        maps = {}
        for mode in (('python', 'cpp') if repeat % 2 == 0 else ('cpp', 'python')):
            start = perf_counter()
            maps[mode] = (python_evidence(inputs, args.resolution) if mode == 'python'
                          else native_evidence(inputs, args.resolution, dll))
            times[mode].append(perf_counter() - start)
        if maps['python'] != maps['cpp']:
            raise ValueError('Evidence differs between backends')
    print(json.dumps(dict(total_scans=len(estimates), accepted_scans=len(inputs),
                          equal_cells=len(maps['python']), observed_seconds=times,
                          speedup=median(times['python']) / median(times['cpp'])), indent=2))


if __name__ == '__main__':
    main()
