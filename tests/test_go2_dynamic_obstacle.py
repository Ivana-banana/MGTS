import numpy as np
import pytest

from tools.go2_dynamic_obstacle import (REQUIRED_SEPARATION_M, build, choose_wait,
                                       closest_encounter, segment_grid_clearance)
from tools.go2_obstacle_demo import Grid


def test_crossing_between_samples_uses_relative_motion():
    times = [0., 1.]
    robot = [[0., 0.], [2., 0.]]
    person = [[1., -1.], [1., 1.]]
    assert closest_encounter(times, robot, times, person) == pytest.approx((0., .5))


def test_blocked_cell_edges_corners_and_map_exterior():
    pixels = np.full((7, 7), 255, dtype=np.uint8)
    pixels[3, 3] = 128  # Unknown occupies the entire closed [3,4] x [3,4] cell.
    grid = Grid(pixels, {'resolution': 1., 'origin': [0., 0.], 'free_thresh': .196})
    assert segment_grid_clearance(grid, [2., 3.5], [5., 3.5]) == 0.
    assert segment_grid_clearance(grid, [2., 2.], [2., 5.]) == pytest.approx(1.)
    assert segment_grid_clearance(grid, [2., 2.], [2., 2.]) == pytest.approx(np.sqrt(2))
    assert segment_grid_clearance(grid, [1., 2.], [-1., 2.]) == 0.


def test_stop_cross_resume_and_round_trip_display_coordinates():
    pixels = np.full((50, 50), 255, dtype=np.uint8)
    grid = Grid(pixels, {'resolution': .1, 'origin': [0., 0.], 'free_thresh': .196})
    frame = [2., -.5, np.pi/2]
    result = build(grid, {'start': [2.4, 4.2], 'goal': [2.4, .8]}, frame)
    timeline = result['timeline']
    time = np.asarray(timeline['t_s'])
    robot, person = np.asarray(timeline['robot_xy']), np.asarray(timeline['obstacle_xy'])
    assert np.all(np.diff(time) > 0)
    assert robot[-1] == pytest.approx([2.4, .8])
    start, end = result['wait']['start_s'], result['wait']['end_s']
    assert np.linalg.norm(np.ptp(robot[(time >= start) & (time <= end)], axis=0)) < 1e-10
    assert result['checks']['baseline_min_center_distance_m'] < 1e-10
    clearance, _ = closest_encounter(time, robot, time, person)
    assert clearance >= REQUIRED_SEPARATION_M
    assert clearance == pytest.approx(result['checks']['planned_min_center_distance_m'])
    transformed = np.column_stack((2 - robot[:, 1], -.5 + robot[:, 0]))
    np.testing.assert_allclose(timeline['robot_xy_map3d'], transformed)
    midpoint = (robot[0] + robot[-1]) / 2
    assert choose_wait(robot[0], robot[-1], [0., 20.], [midpoint, midpoint]) is None
