import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from mgts_nav.bag import load_config, scans
from mgts_nav.demo import PROFILE, generate
from mgts_nav.grid import Grid, read_map
from mgts_nav.metrics import read_csv
from mgts_nav.slam import WallSurface, surface_points, track


def test_neighbourhood_mean_moves_cell_centres_onto_the_wall_line():
    cell = 0.08
    x = (np.arange(-40, 40) + 0.5) * cell
    # A wall at y = 0 lies on a cell border: two rows are occupied at +-half a cell.
    double = np.vstack([np.column_stack([x, np.full_like(x, -cell / 2)]),
                        np.column_stack([x, np.full_like(x, cell / 2)])])
    inner = np.abs(double[:, 0]) < 2.5
    assert np.abs(surface_points(double)[inner, 1]).max() < 1e-9
    # A sloped wall rasterised as a staircase gets closer to its true line.
    xs = (np.arange(-60, 60) + 0.5) * cell
    stairs = np.column_stack([xs, (np.floor(0.3 * xs / cell) + 0.5) * cell])
    before = np.abs(stairs[:, 1] - 0.3 * stairs[:, 0])
    after = surface_points(stairs)
    after = np.abs(after[:, 1] - 0.3 * after[:, 0])
    middle = slice(10, -10)
    assert after[middle].mean() < 0.6 * before[middle].mean()


def test_duplicate_points_are_returned_unchanged():
    points = np.zeros((20, 2))
    np.testing.assert_array_equal(surface_points(points), points)


def test_office_inspection_has_no_half_cell_bias(tmp_path):
    root = tmp_path / "demo"
    generate(root)
    cfg = load_config(PROFILE)
    grid = Grid(0.08)
    for _ in track(scans(root / "mapping", cfg), np.zeros(3), grid=grid):
        pass
    grid.save(tmp_path)
    points = read_map(tmp_path / "map.yaml")[2]
    initial = json.loads((root / "initial_pose.json").read_text(encoding="utf-8"))
    rows = list(track(scans(root / "inspection", cfg), initial, map_points=points, surface=True))
    truth = np.array([[p.x, p.y] for p in read_csv(root / "inspection_truth.csv")])
    error = np.array([r.pose[:2] for r in rows]) - truth
    # Cell-centre matching kept a constant offset of about 3.5 cm along y.
    assert all(r.status == "matched" for r in rows)
    assert np.abs(error.mean(axis=0)).max() < 0.01
    assert np.sqrt((error ** 2).sum(axis=1).mean()) < 0.01


def test_projection_keeps_wall_ends_and_clutter_in_place():
    cell = 0.08
    x = (np.arange(0, 20) + 0.5) * cell
    # The wall ends at x = 0 and 1.6 m; the 3x3 mean would pull the end cells inward.
    wall = np.vstack([np.column_stack([x, np.full_like(x, -cell / 2)]),
                      np.column_stack([x, np.full_like(x, cell / 2)])])
    out = surface_points(wall)
    np.testing.assert_allclose(out[:, 0], wall[:, 0], atol=1e-9)
    inner = (wall[:, 0] > cell) & (wall[:, 0] < x[-1] - cell / 2)
    assert np.abs(out[inner, 1]).max() < 1e-9
    # The 2x2 block at each end is not line-like and keeps its centres.
    np.testing.assert_array_equal(out[~inner], wall[~inner])
    # A filled block (clutter, a pillar) is not line-like: its centre cell stays.
    block = (np.argwhere(np.ones((5, 5))) + 0.5) * cell
    np.testing.assert_array_equal(surface_points(block)[12], block[12])


def test_surface_is_used_only_when_walls_pin_both_directions():
    cell = 0.08
    along = (np.arange(-60, 60) + 0.5) * cell
    corridor = np.vstack([np.column_stack([along, np.full_like(along, -1.5 + cell / 2)]),
                          np.column_stack([along, np.full_like(along, 1.5 - cell / 2)])])
    across = (np.arange(-18, 18) + 0.5) * cell
    room = np.vstack([corridor[np.abs(corridor[:, 0]) < 1.5],
                      np.column_stack([np.full_like(across, -1.5 + cell / 2), across]),
                      np.column_stack([np.full_like(across, 1.5 - cell / 2), across])])
    for points, expected in ((corridor, False), (room, True)):
        surface = WallSurface(cKDTree(points))
        # A dense scan of the same walls from the middle, in the map frame.
        assert surface.constrains(points + 0.01, np.zeros(3)) is expected


@pytest.mark.parametrize("surface", [False, True])
def test_rotated_office_gains_from_the_surface(tmp_path, surface):
    """Walls at 17 degrees to the grid: no border alignment, still a real gain."""
    root = tmp_path / "demo"
    generate(root)
    cfg = load_config(PROFILE)
    yaw = np.radians(17)
    turn = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    grid = Grid(0.08)
    for _ in track(scans(root / "mapping", cfg), np.array([0, 0, yaw]), grid=grid):
        pass
    grid.save(tmp_path)
    points = read_map(tmp_path / "map.yaml")[2]
    initial = json.loads((root / "initial_pose.json").read_text(encoding="utf-8"))
    start = np.r_[turn @ initial[:2], initial[2] + yaw]
    rows = list(track(scans(root / "inspection", cfg), start, map_points=points, surface=surface))
    truth = np.array([turn @ [p.x, p.y] for p in read_csv(root / "inspection_truth.csv")])
    error = np.array([r.pose[:2] for r in rows]) - truth
    ate = np.sqrt((error ** 2).sum(axis=1).mean())
    # 10 seeds: cell centres 1.13 +- 0.10 cm, wall surface 0.68 +- 0.09 cm.
    assert (ate < 0.009) if surface else (ate > 0.009)
