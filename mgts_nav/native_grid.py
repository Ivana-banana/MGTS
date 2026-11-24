"""Optional C++ ray rasterizer. The standard mapping path needs no DLL."""
import ctypes
from pathlib import Path
from weakref import finalize

import numpy as np

from .grid import Grid


def load_dll(path):
    path = Path(path).resolve()
    try:
        dll = ctypes.CDLL(str(path))
        handle, size = ctypes.c_void_p, ctypes.c_size_t
        array = np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS')
        for name, result, args in (
            ('grid_create', handle, []), ('grid_destroy', None, [handle]),
            ('grid_observe', ctypes.c_int, [handle, array, array, size]),
            ('grid_size', size, [handle]), ('grid_export', ctypes.c_int, [handle, array, size]),
        ):
            function = getattr(dll, name)
            function.restype, function.argtypes = result, args
    except (OSError, AttributeError) as exc:
        raise ValueError(f'Cannot load MGTS native grid from {path}: {exc}') from exc
    return dll


def cells32(points, resolution):
    cells = np.floor(points / resolution)
    if not ((cells >= -2**31) & (cells < 2**31)).all():
        raise ValueError('Cell coordinate outside int32 range')
    return np.ascontiguousarray(cells, dtype=np.int32)


class NativeGrid:
    def __init__(self, resolution, dll):
        self.output = Grid(resolution)
        self.dll = dll
        self.handle = dll.grid_create()
        if not self.handle:
            raise ValueError('Cannot allocate native occupancy grid')
        self._release = finalize(self, dll.grid_destroy, self.handle)

    def observe(self, origin, endpoints):
        resolution = self.output.resolution
        near = endpoints[np.linalg.norm(endpoints - origin, axis=1) <= 35]
        cells = np.unique(cells32(near, resolution), axis=0)
        if self.dll.grid_observe(self.handle, cells32(origin, resolution), cells, len(cells)):
            raise ValueError('Cannot update native occupancy grid')

    def evidence(self):
        rows = np.empty((self.dll.grid_size(self.handle), 3), dtype=np.int32)
        if self.dll.grid_export(self.handle, rows, len(rows)):
            raise ValueError('Cannot export native occupancy grid')
        return {(x, y): value for x, y, value in rows.tolist()}

    def save(self, directory):
        self.output.evidence = self.evidence()
        self.output.save(directory)
