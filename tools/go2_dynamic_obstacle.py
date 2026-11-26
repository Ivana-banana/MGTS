"""Schedule a Go2 crossing on the published Go2 raster with a moving pedestrian."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mgts_nav.grid import read_map
from tools.go2_obstacle_demo import Grid, ROBOT_RADIUS_M, SAFETY_MARGIN_M, to_frame


MAP_SHA256 = '219c86b854a88c7ce7b8fbb391ac4a38e2e5e6464e54ae4cd25a8e074d9c6666'
PERSON_RADIUS_M = .22
PERSON_SPEED_M_S = .40
ROBOT_SPEED_M_S = .45
REQUIRED_SEPARATION_M = ROBOT_RADIUS_M + PERSON_RADIUS_M + SAFETY_MARGIN_M


def interpolate(times, xy, at):
    """Positions outside the time knots remain at the corresponding endpoint."""
    return np.column_stack([np.interp(at, times, axis) for axis in np.asarray(xy).T])


def closest_encounter(robot_t, robot_xy, person_t, person_xy):
    """Exact minimum between two piecewise linear center trajectories."""
    times = np.union1d(robot_t, person_t)
    relative = interpolate(robot_t, robot_xy, times) - interpolate(person_t, person_xy, times)
    delta = np.diff(relative, axis=0)
    squared_speed = np.sum(delta**2, axis=1)
    along = np.divide(-np.sum(relative[:-1] * delta, axis=1), squared_speed,
                      out=np.zeros_like(squared_speed), where=squared_speed > 0)
    along = np.clip(along, 0, 1)
    distances = np.linalg.norm(relative[:-1] + along[:, None] * delta, axis=1)
    i = int(np.argmin(distances))
    return float(distances[i]), float(times[i] + along[i] * (times[i+1] - times[i]))


def segment_grid_clearance(grid, start, end):
    """Exact centerline distance to closed blocked cell squares and the map exterior."""
    start, end = np.asarray(start), np.asarray(end)
    step = end - start
    bounds = grid.origin + grid.res * np.array([grid.width, grid.height])
    outside_distance = max(0., float(np.min([start - grid.origin, end - grid.origin,
                                             bounds - start, bounds - end])))
    if outside_distance == 0:
        return 0.
    rows, cols = np.nonzero(~grid.free)
    if not len(rows):
        return outside_distance
    lo = grid.origin + np.column_stack((cols, grid.height - 1 - rows)) * grid.res
    hi = lo + grid.res

    # A crossing through two edges can have positive endpoint/corner distances.
    enter, leave = np.zeros(len(rows)), np.ones(len(rows))
    overlaps = np.ones(len(rows), dtype=bool)
    for k in (0, 1):
        if abs(step[k]) < 1e-14:
            overlaps &= (lo[:, k] <= start[k]) & (start[k] <= hi[:, k])
        else:
            first = (lo[:, k] - start[k]) / step[k]
            last = (hi[:, k] - start[k]) / step[k]
            enter = np.maximum(enter, np.minimum(first, last))
            leave = np.minimum(leave, np.maximum(first, last))
    if np.any(overlaps & (enter <= leave)):
        return 0.

    nearest = outside_distance
    for endpoint in (start, end):
        gaps = np.maximum(np.maximum(lo-endpoint, endpoint-hi), 0.)
        nearest = min(nearest, float(np.linalg.norm(gaps, axis=1).min()))
    length_squared = step @ step
    if length_squared:
        for dx, dy in ((0, 0), (0, 1), (1, 0), (1, 1)):
            corners = lo + grid.res * np.array([dx, dy])
            along = np.clip(((corners - start) @ step) / length_squared, 0., 1.)
            nearest = min(nearest, float(np.linalg.norm(corners-start-along[:, None]*step,
                                                        axis=1).min()))
    return nearest


def robot_path(start, goal, wait):
    unit = (goal - start) / np.linalg.norm(goal - start)
    stop = (start + goal) / 2 - .90 * unit
    stop_t = np.linalg.norm(stop - start) / ROBOT_SPEED_M_S
    finish = np.linalg.norm(goal - start) / ROBOT_SPEED_M_S
    if wait == 0:
        return np.array([0., finish]), np.array([start, goal])
    return np.array([0., stop_t, stop_t + wait, finish + wait]), np.array([start, stop, stop, goal])


def choose_wait(start, goal, person_t, person_xy):
    """First admissible stop dwell on a 0.10 s schedule, up to 12 s."""
    for tick in range(121):
        wait = tick / 10
        robot_t, robot_xy = robot_path(start, goal, wait)
        separation, _ = closest_encounter(robot_t, robot_xy, person_t, person_xy)
        if separation >= REQUIRED_SEPARATION_M:
            return wait
    return None


def build(grid, segment, frame):
    start, goal = np.asarray(segment['start']), np.asarray(segment['goal'])
    length = np.linalg.norm(goal - start)
    if length <= 1.80:
        raise ValueError('Selected route is too short for the scheduled waiting point')
    heading = (goal - start) / length
    normal = np.array([-heading[1], heading[0]])
    center = (start + goal) / 2
    extent = .83
    person_xy = np.array([center - extent*normal, center - extent*normal, center + extent*normal])
    crossing_t = length / (2 * ROBOT_SPEED_M_S)
    person_t = np.array([0., crossing_t - extent/PERSON_SPEED_M_S, crossing_t + extent/PERSON_SPEED_M_S])
    if person_t[1] <= 0:
        raise ValueError('Pedestrian starts before the scenario clock')

    wall_robot = segment_grid_clearance(grid, start, goal)
    wall_person = segment_grid_clearance(grid, person_xy[0], person_xy[-1])
    if wall_robot < ROBOT_RADIUS_M + SAFETY_MARGIN_M or wall_person < PERSON_RADIUS_M + SAFETY_MARGIN_M:
        raise ValueError('Robot or pedestrian sweep intersects the blocked grid or required margin')

    direct_t, direct_xy = robot_path(start, goal, 0)
    baseline, baseline_t = closest_encounter(direct_t, direct_xy, person_t, person_xy)
    wait = choose_wait(start, goal, person_t, person_xy)
    if baseline >= ROBOT_RADIUS_M + PERSON_RADIUS_M or not wait:
        raise ValueError('Expected a baseline collision and a nonzero feasible dwell')
    robot_t, robot_xy = robot_path(start, goal, wait)
    separation, nearest_t = closest_encounter(robot_t, robot_xy, person_t, person_xy)
    stuck_wait = choose_wait(start, goal, [0., robot_t[-1]], np.array([center, center]))
    if stuck_wait is not None:
        raise ValueError('A permanently blocked crossing must reject every departure')

    # Union of camera samples and exact velocity change times. Linear interpolation
    # in the exported timeline reproduces the checked robot and person paths.
    times = np.unique(np.r_[np.arange(0., robot_t[-1], .1), robot_t, person_t])
    robot = interpolate(robot_t, robot_xy, times)
    person = interpolate(person_t, person_xy, times)
    yaw = float(np.arctan2(heading[1], heading[0]))
    person_yaw = float(np.arctan2(normal[1], normal[0]))
    return {
        'scenario': 'modeled_pedestrian_crossing', 'action': 'yield_then_continue',
        'frames': {'planning': 'map2d', 'display': 'map3d',
                   'map2d_origin_in_map3d_xy_yaw': list(frame)},
        'map': {'pgm_sha256': MAP_SHA256, 'resolution_m': grid.res,
                'origin': grid.origin.tolist(), 'unknown_is_blocked': True},
        'robot': {'radius_m': ROBOT_RADIUS_M, 'moving_speed_m_s': ROBOT_SPEED_M_S},
        'obstacle': {'type': 'modeled_pedestrian', 'footprint': 'disc',
                     'radius_m': PERSON_RADIUS_M, 'walking_speed_m_s': PERSON_SPEED_M_S},
        'safety_margin_m': SAFETY_MARGIN_M,
        'wait': {'start_s': float(robot_t[1]), 'end_s': float(robot_t[2]), 'duration_s': wait},
        'duration_s': float(times[-1]),
        'checks': {
            'baseline_min_center_distance_m': baseline, 'baseline_conflict_time_s': baseline_t,
            'planned_min_center_distance_m': separation, 'planned_closest_time_s': nearest_t,
            'required_center_distance_m': REQUIRED_SEPARATION_M,
            'planned_min_body_to_person_gap_m': separation - ROBOT_RADIUS_M - PERSON_RADIUS_M,
            'min_robot_body_to_blocked_grid_m': wall_robot - ROBOT_RADIUS_M,
            'min_person_body_to_blocked_grid_m': wall_person - PERSON_RADIUS_M,
            'permanently_blocked_result': 'no_safe_wait_on_this_path',
        },
        'timeline': {'t_s': times.tolist(), 'robot_xy': robot.tolist(), 'obstacle_xy': person.tolist(),
                     'robot_yaw_rad': yaw, 'obstacle_yaw_rad': person_yaw,
                     'robot_xy_map3d': to_frame(robot, frame).tolist(),
                     'obstacle_xy_map3d': to_frame(person, frame).tolist(),
                     'robot_yaw_map3d_rad': yaw + frame[2], 'obstacle_yaw_map3d_rad': person_yaw + frame[2]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('map_yaml', type=Path)
    parser.add_argument('static_route', type=Path, help='route JSON from the Go2 fixed obstacle case')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    pixels, meta, _ = read_map(args.map_yaml)
    pgm = args.map_yaml.parent / meta['image']
    if hashlib.sha256(pgm.read_bytes()).hexdigest() != MAP_SHA256:
        parser.error('This scenario expects the published 0.08 m official mapping raster')
    source = json.loads(args.static_route.read_text(encoding='utf-8'))
    if (meta['resolution'] != source['map']['resolution_m'] or meta['origin'][2] != 0
            or list(meta['origin'][:2]) != source['map']['origin']):
        parser.error('Map geometry disagrees with the source route')
    result = build(Grid(pixels, meta), source['segment'], source['map3d_transform']['map2d_origin_in_map3d_xy_yaw'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n',
                        encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('action', 'duration_s', 'wait', 'checks')}, indent=2))


if __name__ == '__main__':
    main()
