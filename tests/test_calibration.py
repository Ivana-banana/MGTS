from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mgts_nav.calibration import collect, extrinsics, fit_floor


def edge(parent, child, x, angle=0):
    q = Rotation.from_euler('z', angle).as_quat()
    transform = NS(translation=NS(x=x, y=0, z=0), rotation=NS(**dict(zip('xyzw', q))))
    return NS(header=NS(frame_id=parent), child_frame_id=child, transform=transform)


def test_rotated_mount_and_different_tree_branch():
    edges = {}
    for e in [edge('office', 'base', 2), edge('base', 'mount', 1, np.pi / 2),
              edge('mount', 'lidar', 2), edge('office', 'imu', 3)]:
        collect(edges, e)
    result = extrinsics(edges, 'base', 'lidar', 'imu')
    assert result['base_from_lidar_xyz_m'] == pytest.approx([1, 2, 0])
    assert result['base_from_imu_xyzw'] == pytest.approx([0, 0, 0, 1])


def test_conflicting_or_missing_static_transform():
    edges = {}
    collect(edges, edge('base', 'lidar', 2))
    collect(edges, edge('base', 'lidar', 2))
    with pytest.raises(ValueError, match='Conflicting'):
        collect(edges, edge('base', 'lidar', 3))
    with pytest.raises(ValueError, match='No static TF path'):
        extrinsics(edges, 'base', 'lidar', 'imu')
    assert extrinsics(edges, 'base', 'lidar')['base_from_lidar_xyz_m'] == [2, 0, 0]


def test_floor_tilt_requires_two_dimensional_support():
    x, y = np.meshgrid(np.linspace(-4, 4, 20), np.linspace(-3, 3, 18))
    cloud = np.column_stack([x.ravel(), y.ravel(), (.1*x+.2*y).ravel()])
    result = fit_floor(cloud)
    assert result['floor_fraction'] == 1
    assert result['floor_rise_pitch_deg'] == pytest.approx(np.degrees(np.arctan(.1)))
    cloud[:, 1] = 0
    with pytest.raises(ValueError, match='constrain'):
        fit_floor(cloud)
