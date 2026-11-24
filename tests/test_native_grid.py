import shutil
import subprocess
import sys

import pytest

from mgts_nav.demo import PROFILE, generate
from mgts_nav.native_grid import load_dll
from mgts_nav.solve import solve
from tools.benchmark_native_grid import verify_edge_cases


def test_missing_dll_reports_requested_file(tmp_path):
    with pytest.raises(ValueError, match=r'Cannot load MGTS native grid.*absent\.dll'):
        load_dll(tmp_path / 'absent.dll')


@pytest.mark.skipif(sys.platform != 'win32' or not shutil.which('cl'),
                    reason='Native integration requires a Windows MSVC developer shell')
def test_compiled_grid_matches_reference_solve(tmp_path):
    dll = tmp_path / 'grid.dll'
    subprocess.run(['cl', '/nologo', '/O2', '/EHsc', '/std:c++17', '/LD', 'tools/native_grid.cpp',
                    f'/Fo:{tmp_path / "grid.obj"}', f'/Fe:{dll}'], check=True)
    verify_edge_cases(load_dll(dll))
    data = tmp_path / 'bag'
    generate(data)
    for mode in ('python', 'cpp'):
        solve(data / 'mapping', data / 'inspection', PROFILE, tmp_path / mode,
              close_loop=True, grid_dll=dll if mode == 'cpp' else None)
    for name in ('map/map.pgm', 'map/map.yaml', 'map/trajectory.csv', 'map/loop.json',
                 'inspection/trajectory.csv', 'inspection/continuous.csv'):
        assert (tmp_path / 'python' / name).read_bytes() == (tmp_path / 'cpp' / name).read_bytes(), name
