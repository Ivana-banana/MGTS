import numpy as np

from tools.go2_obstacle_demo import Grid, astar, box_distance, clearance_map, path_check


def corridor(width_m=3.0, length_m=8.0, res=0.08):
    """Free corridor along y with occupied walls and unknown space outside."""
    cols = int(round((width_m + 1.0) / res))
    rows = int(round(length_m / res))
    pixels = np.full((rows, cols), 205, np.uint8)
    pixels[:, int(0.5 / res):int((0.5 + width_m) / res)] = 254
    pixels[:, int(0.5 / res) - 1] = 0
    pixels[:, int((0.5 + width_m) / res)] = 0
    meta = {'resolution': res, 'origin': [-0.5 - width_m / 2, 0.0, 0.0], 'free_thresh': 0.196,
            'occupied_thresh': 0.65}
    return Grid(pixels, meta)


def test_detour_avoids_box_and_walls():
    grid = corridor()
    start, goal, radius = np.array([0.0, 1.0]), np.array([0.0, 7.0]), 0.48
    base = astar(grid, clearance_map(grid), start, goal, radius)
    box = (0.0, 4.0, 0.5, 0.5)
    assert path_check(grid, base, [box])['intersects_obstacle_0']
    detour = astar(grid, clearance_map(grid, [box]), start, goal, radius)
    check = path_check(grid, detour, [box])
    assert not check['intersects_obstacle_0']
    assert check['min_obstacle_0_clearance_m'] >= radius - grid.res / 2
    assert check['min_wall_or_unknown_clearance_m'] >= radius - grid.res / 2
    assert check['length_m'] < 1.5 * path_check(grid, base)['length_m']


def test_blocked_corridor_and_unknown_space_are_refused():
    grid = corridor()
    start, goal, radius = np.array([0.0, 1.0]), np.array([0.0, 7.0]), 0.48
    wall = (0.0, 4.0, 12.0, 0.5)
    assert astar(grid, clearance_map(grid, [wall]), start, goal, radius) is None
    # A corridor narrower than the robot disc is refused even without an obstacle.
    narrow = corridor(width_m=0.8)
    assert astar(narrow, clearance_map(narrow), start, goal, radius) is None


def test_box_distance():
    x, y = np.array([0.0, 1.0, 1.0]), np.array([0.0, 0.0, 1.0])
    assert np.allclose(box_distance(x, y, (0, 0, 1, 1)), [0, 0.5, np.hypot(0.5, 0.5)])
