import json

import numpy as np

from mgts_nav.bag import load_config, scans
from mgts_nav.demo import PROFILE, generate
from mgts_nav.grid import Grid, read_map
from mgts_nav.metrics import read_csv
from mgts_nav.slam import surface_points, track


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
    rows = list(track(scans(root / "inspection", cfg), initial, map_points=points))
    truth = np.array([[p.x, p.y] for p in read_csv(root / "inspection_truth.csv")])
    error = np.array([r.pose[:2] for r in rows]) - truth
    # Cell-centre matching kept a constant offset of about 3.5 cm along y.
    assert all(r.status == "matched" for r in rows)
    assert np.abs(error.mean(axis=0)).max() < 0.01
    assert np.sqrt((error ** 2).sum(axis=1).mean()) < 0.01
