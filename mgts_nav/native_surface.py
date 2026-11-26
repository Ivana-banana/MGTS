"""Optional C++ per-scan core of the 3D tracker (#190). The tracker needs no DLL.

`NativeSurface(surface, dll)` takes a built SurfaceMap (normals and planar flags stay
computed in Python, once) and offers the same `align` and `score` backed by
tools/native_surface.cpp.
"""
import ctypes
from pathlib import Path
from weakref import finalize

import numpy as np
from scipy.spatial.transform import Rotation

from .localize3d import MAX_MATCH_M, Pose


def load_dll(path):
    path = Path(path).resolve()
    try:
        dll = ctypes.CDLL(str(path))
        doubles = np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS')
        flags = np.ctypeslib.ndpointer(np.uint8, flags='C_CONTIGUOUS')
        handle, size = ctypes.c_void_p, ctypes.c_size_t
        for name, result, args in (
            ('surf_create', handle, [doubles, doubles, flags, size, doubles, doubles, size, ctypes.c_double]),
            ('surf_destroy', None, [handle]),
            ('surf_align', ctypes.c_int, [handle, doubles, size, doubles, doubles, ctypes.POINTER(ctypes.c_int)]),
            ('surf_score', ctypes.c_double, [handle, doubles, size, doubles]),
        ):
            function = getattr(dll, name)
            function.restype, function.argtypes = result, args
    except (OSError, AttributeError) as exc:
        raise ValueError(f'Cannot load MGTS native surface from {path}: {exc}') from exc
    return dll


def _doubles(values):
    return np.ascontiguousarray(values, dtype=np.float64)


def _pose(pose):
    return _doubles(np.r_[pose.rotation.as_quat(), pose.xyz])


class NativeSurface:
    def __init__(self, surface, dll, radius=MAX_MATCH_M):
        self.dll = dll
        self.handle = dll.surf_create(
            _doubles(surface.tree.data), _doubles(surface.normals),
            np.ascontiguousarray(surface.planar, dtype=np.uint8), surface.tree.n,
            _doubles(surface.plane_tree.data.reshape(-1, 3)), _doubles(surface.plane_normals.reshape(-1, 3)),
            surface.plane_tree.n, radius)
        if not self.handle:
            raise ValueError('Cannot build native surface map')
        self._release = finalize(self, dll.surf_destroy, self.handle)

    def align(self, points, initial):
        points = _doubles(points)
        out = np.empty(7)
        accepted = ctypes.c_int()
        code = self.dll.surf_align(self.handle, points, len(points), _pose(initial), out, ctypes.byref(accepted))
        if code:
            raise ValueError(f'Native alignment failed with code {code}')
        if not accepted.value:
            return initial, False
        return Pose(Rotation.from_quat(out[:4]), out[4:]), True

    def score(self, points, pose):
        points = _doubles(points)
        return float(self.dll.surf_score(self.handle, points, len(points), _pose(pose)))
