"""Model scenario: detour around a test obstacle on the official Go2 mapping grid (#212).

This is a planning demonstration on a real part of the map, not a result of the
robot.  The mapping grid gives occupied, free and unknown cells.  Go2 is a disc
of ROBOT_RADIUS_M (the circumscribed circle of a 0.70 x 0.31 m body) plus
SAFETY_MARGIN_M; the planner only uses observed free cells, so unknown space counts
as blocked.

Steps:
1. pick start and goal on the recorded mapping route, SEGMENT_M apart along it; among
   segments whose straight line clears the robot disc and whose corridor is wide enough,
   take the roomiest one (largest line clearance) with a detour <= MAX_DETOUR_RATIO;
2. plan the route without the obstacle (A*, 8-connected grid);
3. put an axis-aligned test box of OBSTACLE_SIZE_M on the middle of that route and
   plan again;
4. control: widen the box across the corridor; the planner must refuse.

Every route is checked against walls, unknown cells and the inflated box, and the
minimum clearance along it is reported.
"""
import argparse
import csv
import heapq
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from mgts_nav.grid import read_map

ROBOT_RADIUS_M = 0.38
SAFETY_MARGIN_M = 0.10
OBSTACLE_SIZE_M = (0.5, 0.5)
SEGMENT_M = 4.0
MIN_DETOUR_WIDTH_M = 2 * (ROBOT_RADIUS_M + SAFETY_MARGIN_M) + 0.5 * max(OBSTACLE_SIZE_M) + 0.3
BLOCK_WIDTH_M = 12.0
MAX_DETOUR_RATIO = 1.5
CANDIDATE_STEP = 5
FREE, UNKNOWN, OCCUPIED = 'free', 'unknown', 'occupied'


class Grid:
    def __init__(self, pixels, meta):
        self.res = float(meta['resolution'])
        self.origin = np.asarray(meta['origin'][:2], float)
        occupancy = 1.0 - pixels / 255.0
        self.free = occupancy < meta['free_thresh']
        self.height, self.width = pixels.shape
        self.blocked_distance = distance_transform_edt(self.free) * self.res

    def cell(self, xy):
        col = int(np.floor((xy[0] - self.origin[0]) / self.res))
        row = self.height - 1 - int(np.floor((xy[1] - self.origin[1]) / self.res))
        return row, col

    def centre(self, row, col):
        return self.origin + (np.array([col, self.height - 1 - row]) + 0.5) * self.res

    def centres(self):
        rows, cols = np.mgrid[0:self.height, 0:self.width]
        x = self.origin[0] + (cols + 0.5) * self.res
        y = self.origin[1] + (self.height - 1 - rows + 0.5) * self.res
        return x, y


def box_distance(x, y, box):
    """Distance from points to an axis-aligned box given as (cx, cy, sx, sy)."""
    cx, cy, sx, sy = box
    dx = np.maximum(np.abs(x - cx) - sx / 2, 0)
    dy = np.maximum(np.abs(y - cy) - sy / 2, 0)
    return np.hypot(dx, dy)


def clearance_map(grid, boxes=()):
    """Distance from each cell centre to the nearest wall, unknown cell or box."""
    clearance = grid.blocked_distance.copy()
    x, y = grid.centres()
    for box in boxes:
        clearance = np.minimum(clearance, box_distance(x, y, box))
    return clearance


def astar(grid, clearance, start, goal, radius):
    ok = clearance >= radius
    s, g = grid.cell(start), grid.cell(goal)
    if not (ok[s] and ok[g]):
        return None
    steps = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
             (-1, -1, 2 ** .5), (-1, 1, 2 ** .5), (1, -1, 2 ** .5), (1, 1, 2 ** .5)]
    cost = {s: 0.0}
    parent = {}
    queue = [(0.0, s)]
    while queue:
        _, current = heapq.heappop(queue)
        if current == g:
            path = [current]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            return np.array([grid.centre(*c) for c in path[::-1]])
        for dr, dc, w in steps:
            nxt = (current[0] + dr, current[1] + dc)
            if not (0 <= nxt[0] < grid.height and 0 <= nxt[1] < grid.width) or not ok[nxt]:
                continue
            new = cost[current] + w
            if new < cost.get(nxt, np.inf):
                cost[nxt] = new
                parent[nxt] = current
                heapq.heappush(queue, (new + np.hypot(nxt[0] - g[0], nxt[1] - g[1]), nxt))
    return None


def densify(points, step):
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out)


def path_check(grid, path, boxes=()):
    """Clearance along a polyline sampled at a quarter cell."""
    dense = densify(path, grid.res / 4)
    walls = []
    for p in dense:
        row, col = grid.cell(p)
        inside = 0 <= row < grid.height and 0 <= col < grid.width
        walls.append(grid.blocked_distance[row, col] if inside else 0.0)
    walls = np.array(walls)
    out = {'length_m': round(float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))), 3),
           'min_wall_or_unknown_clearance_m': round(float(walls.min()), 3)}
    for k, box in enumerate(boxes):
        d = box_distance(dense[:, 0], dense[:, 1], box)
        out[f'min_obstacle_{k}_clearance_m'] = round(float(d.min()), 3)
        out[f'intersects_obstacle_{k}'] = bool(d.min() == 0)
    return out


def lateral_width(grid, point, direction):
    """Free width across the route at `point`, measured on observed free cells."""
    normal = np.array([-direction[1], direction[0]])
    width = 0.0
    for sign in (1, -1):
        d = 0.0
        while d < 5:
            row, col = grid.cell(point + sign * (d + grid.res) * normal)
            if not (0 <= row < grid.height and 0 <= col < grid.width) or not grid.free[row, col]:
                break
            d += grid.res
        width += d
    return width


def segment_candidates(grid, route, radius):
    """Straight-enough route segments, roomiest first (largest clearance of the line)."""
    along = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    found = []
    for i in range(0, len(route), CANDIDATE_STEP):
        j = int(np.searchsorted(along, along[i] + SEGMENT_M))
        if j >= len(route):
            break
        a, b = route[i], route[j]
        if np.linalg.norm(b - a) < 0.75 * SEGMENT_M:
            continue
        line = path_check(grid, np.array([a, b]))['min_wall_or_unknown_clearance_m']
        if line < radius:
            continue
        width = lateral_width(grid, (a + b) / 2, (b - a) / np.linalg.norm(b - a))
        if width >= MIN_DETOUR_WIDTH_M:
            found.append((line, width, i, j))
    return [(i, j) for _, _, i, j in sorted(found, reverse=True)]


def scenario(grid, start, goal, radius):
    base = astar(grid, clearance_map(grid), start, goal, radius)
    if base is None:
        return None
    middle = base[len(base) // 2]
    box = (float(middle[0]), float(middle[1]), *OBSTACLE_SIZE_M)
    detour = astar(grid, clearance_map(grid, [box]), start, goal, radius)
    return base, box, detour


def read_route(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return np.array([[float(r['x']), float(r['y'])] for r in csv.DictReader(stream)])


def to_frame(points, origin):
    """2D map frame -> 3D map (shared odometry) frame given the 2D origin pose there."""
    c, s = np.cos(origin[2]), np.sin(origin[2])
    pts = np.asarray(points, float).reshape(-1, 2)
    return np.column_stack([origin[0] + c * pts[:, 0] - s * pts[:, 1], origin[1] + s * pts[:, 0] + c * pts[:, 1]])


def run(map_yaml, route_csv, origin_3d=None):
    pixels, meta, _ = read_map(map_yaml)
    grid = Grid(pixels, meta)
    radius = ROBOT_RADIUS_M + SAFETY_MARGIN_M
    route = read_route(route_csv)
    for i, j in segment_candidates(grid, route, radius):
        start, goal = route[i], route[j]
        found = scenario(grid, start, goal, radius)
        if found is None or found[2] is None:
            continue
        base, box, detour = found
        ratio = path_check(grid, detour)['length_m'] / path_check(grid, base)['length_m']
        if ratio <= MAX_DETOUR_RATIO:
            break
    else:
        raise SystemExit('no segment with a local detour on this route')
    direction = (goal - start) / np.linalg.norm(goal - start)
    # Control: the same box stretched across the corridor, perpendicular to the route.
    if abs(direction[0]) > abs(direction[1]):
        wall = (box[0], box[1], OBSTACLE_SIZE_M[0], BLOCK_WIDTH_M)
    else:
        wall = (box[0], box[1], BLOCK_WIDTH_M, OBSTACLE_SIZE_M[1])
    blocked = astar(grid, clearance_map(grid, [wall]), start, goal, radius)
    straight = np.array([start, goal])
    result = {
        'label': 'Модельный сценарий: тестовое препятствие на карте обхода, не наблюдение робота',
        'frame': 'map2d: frame of the 2D mapping grid (origin = first mapping scan)',
        'map': {'yaml': str(map_yaml), 'resolution_m': grid.res, 'origin': grid.origin.tolist(),
                'unknown_is_blocked': True},
        'robot': {'model': 'Go2 body 0.70 x 0.31 m as circumscribed disc', 'radius_m': ROBOT_RADIUS_M,
                  'safety_margin_m': SAFETY_MARGIN_M, 'planning_radius_m': radius},
        'segment': {'route_csv': str(route_csv), 'start_index': int(i), 'goal_index': int(j),
                    'start': start.tolist(), 'goal': goal.tolist()},
        'obstacle': {'shape': 'axis-aligned box', 'center': [box[0], box[1]], 'size_m': list(OBSTACLE_SIZE_M)},
        'control_obstacle': {'center': [wall[0], wall[1]], 'size_m': [wall[2], wall[3]],
                             'result': 'refused' if blocked is None else 'path found'},
        'routes': {
            'straight': {'xy': straight.tolist(), 'check': path_check(grid, straight, [box])},
            'original': {'xy': base.tolist(), 'check': path_check(grid, base, [box])},
            'detour': None if detour is None else {'xy': detour.tolist(), 'check': path_check(grid, detour, [box])},
        },
    }
    checks = {
        'original_crosses_obstacle': result['routes']['original']['check']['intersects_obstacle_0'],
        'straight_crosses_obstacle': result['routes']['straight']['check']['intersects_obstacle_0'],
        'detour_found': detour is not None,
        'detour_keeps_planning_radius_from_walls_and_unknown':
            detour is not None and result['routes']['detour']['check']['min_wall_or_unknown_clearance_m'] >= radius - grid.res / 2,
        'detour_keeps_planning_radius_from_obstacle':
            detour is not None and result['routes']['detour']['check']['min_obstacle_0_clearance_m'] >= radius - grid.res / 2,
        'blocked_corridor_refused': blocked is None,
    }
    result['checks'] = checks
    if origin_3d is not None:
        result['map3d_transform'] = {'map2d_origin_in_map3d_xy_yaw': list(map(float, origin_3d)),
                                     'note': 'map3d = shared odometry frame of the a5ad606 3D map'}
        for name, value in result['routes'].items():
            if value is not None:
                value['xy_map3d'] = to_frame(value['xy'], origin_3d).tolist()
        result['obstacle']['center_map3d'] = to_frame(box[:2], origin_3d)[0].tolist()
        result['obstacle']['yaw_in_map3d_rad'] = float(origin_3d[2])
    return result, grid


def plot(result, grid, pixels, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    xs = [p[0] for r in result['routes'].values() if r for p in r['xy']]
    ys = [p[1] for r in result['routes'].values() if r for p in r['xy']]
    pad = 2.0
    extent = [grid.origin[0], grid.origin[0] + grid.width * grid.res,
              grid.origin[1], grid.origin[1] + grid.height * grid.res]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(pixels, cmap='gray', extent=extent, vmin=0, vmax=255)
    o = result['obstacle']
    ax.add_patch(Rectangle((o['center'][0] - o['size_m'][0] / 2, o['center'][1] - o['size_m'][1] / 2),
                           *o['size_m'], color='#d1495b', label='тестовое препятствие'))
    styles = {'original': ('#00798c', '--', 'путь без препятствия'), 'detour': ('#edae49', '-', 'объезд')}
    for name, (color, line, label) in styles.items():
        r = result['routes'][name]
        if r:
            xy = np.array(r['xy'])
            ax.plot(xy[:, 0], xy[:, 1], line, color=color, lw=2, label=label)
    s, g = result['segment']['start'], result['segment']['goal']
    ax.plot(*s, 'o', color='#30a14e', label='старт')
    ax.plot(*g, 's', color='#d1495b', mfc='none', label='цель')
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_title('Модельный сценарий объезда, карта обхода Go2')
    ax.set_xlabel('x, м (кадр 2D-карты)')
    ax.set_ylabel('y, м')
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('map_yaml', type=Path, help='2D map.yaml of the official mapping run')
    parser.add_argument('route_csv', type=Path, help='mapping trajectory.csv in the same 2D map frame')
    parser.add_argument('--mapping', type=Path, help='mapping bag: adds coordinates in the 3D map frame')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    origin = None
    if args.mapping:
        from tools.go2_independent_check import map_origin
        origin = map_origin(args.mapping)[0]
    result, grid = run(args.map_yaml, args.route_csv, origin)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'route.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    plot(result, grid, read_map(args.map_yaml)[0], args.out / 'route.png')
    print(json.dumps({k: result[k] for k in ('segment', 'obstacle', 'control_obstacle', 'checks')}
                     | {n: (r['check'] if r else None) for n, r in result['routes'].items()},
                     indent=1, ensure_ascii=False))
    if not all(result['checks'].values()):
        raise SystemExit('a scenario check failed, see route.json')


if __name__ == '__main__':
    main()
