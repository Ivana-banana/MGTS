"""Report observed occupancy of named map zones from a localized trajectory."""
import argparse
import csv
from dataclasses import dataclass
from fractions import Fraction
import json
from math import hypot, isfinite
from pathlib import Path
import sys


COLUMNS = ('zone', 'first_inside_t', 'last_inside_t', 'entry_observed', 'exit_observed', 'samples')


def turn(a, b, c):
    return (b[0]-a[0]) * (c[1]-a[1]) - (b[1]-a[1]) * (c[0]-a[0])


def intersect(a, b, c, d):
    for axis in (0, 1):
        if max(a[axis], b[axis]) < min(c[axis], d[axis]) or max(c[axis], d[axis]) < min(a[axis], b[axis]):
            return False
    return turn(a, b, c) * turn(a, b, d) <= 0 and turn(c, d, a) * turn(c, d, b) <= 0


def validate_ring(name, vertices):
    # Test the topology of the parsed coordinates without rounding a near miss into a crossing.
    points = [(Fraction(x), Fraction(y)) for x, y in vertices]
    if points[-1] == points[0]:
        points.pop()
    if len(points) < 3 or len(set(points)) != len(points):
        raise ValueError(f'Too few or repeated vertices in zone {name!r}')
    edges = list(zip(points, points[1:] + points[:1]))
    for i, (a, b) in enumerate(edges):
        c = edges[(i+1) % len(edges)][1]
        if turn(a, b, c) == 0 and (a[0]-b[0])*(c[0]-b[0]) + (a[1]-b[1])*(c[1]-b[1]) > 0:
            raise ValueError(f'Overlapping adjacent edges in zone {name!r}')
        for j in range(i+2, len(edges)):
            if i == 0 and j == len(edges)-1:
                continue  # First and last edges share their endpoint.
            if intersect(a, b, *edges[j]):
                raise ValueError(f'Self-intersection in zone {name!r} at edges {i+1} and {j+1}')
    if abs(sum(ax * by - bx * ay for (ax, ay), (bx, by) in edges)) < 1e-12:
        raise ValueError(f'Degenerate polygon: {name!r}')


def load_zones(path):
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict) or data.get('frame') != 'map':
        raise ValueError('Expected polygons in the map frame')
    zones = data.get('zones')
    if not isinstance(zones, dict) or not zones:
        raise ValueError('Expected a nonempty zones object')
    for name, vertices in zones.items():
        if not name.strip() or not isinstance(vertices, list) or len(vertices) < 3:
            raise ValueError(f'Invalid polygon: {name!r}')
        for point in vertices:
            if not isinstance(point, list) or len(point) != 2 or not all(
                type(v) in (float, int) and isfinite(v) for v in point
            ):
                raise ValueError(f'Invalid vertex in {name!r}')
        validate_ring(name, vertices)
    return zones


def inside(x, y, vertices):
    """Even-odd containment; include points on an edge (one nanometre tolerance)."""
    result = False
    for (ax, ay), (bx, by) in zip(vertices, vertices[1:] + vertices[:1]):
        dx, dy = bx - ax, by - ay
        length = hypot(dx, dy)
        if not length:
            continue
        cross = dx * (y - ay) - dy * (x - ax)
        along = dx * (x - ax) + dy * (y - ay)
        if abs(cross) <= 1e-9 * length and -1e-9 <= along / length <= length + 1e-9:
            return True
        if (ay > y) != (by > y) and x < ax + (y - ay) * dx / dy:
            result = not result
    return result


@dataclass
class Visit:
    first: float
    last: float
    entry: bool
    samples: int = 1


def visits(rows, zones, max_gap):
    if not isfinite(max_gap) or max_gap <= 0:
        raise ValueError('max-gap must be positive and finite')
    previous_t = None
    previous_matched = False
    active = {}
    report = []

    def finish(name, observed_exit):
        visit = active.pop(name)
        if visit.samples >= 2:
            report.append((name, visit.first, visit.last, int(visit.entry),
                           int(observed_exit), visit.samples))

    for row in rows:
        t, x, y = (float(row[key]) for key in ('t', 'x', 'y'))
        if not all(map(isfinite, (t, x, y))) or (previous_t is not None and t <= previous_t):
            raise ValueError('Expected finite poses and strictly increasing timestamps')
        matched = row['status'] == 'matched'
        continuous = matched and previous_matched and t - previous_t <= max_gap + 1e-9
        if not continuous:
            for name in list(active):
                finish(name, False)
        if matched:
            for name, polygon in zones.items():
                if inside(x, y, polygon):
                    if name in active:
                        active[name].last = t
                        active[name].samples += 1
                    else:
                        active[name] = Visit(t, t, continuous)
                elif name in active:
                    finish(name, True)
        previous_t, previous_matched = t, matched
    for name in list(active):
        finish(name, False)
    return sorted(report, key=lambda row: (row[1], row[0]))


def read_visits(trajectory, zones, max_gap):
    with trajectory.open(encoding='utf-8-sig', newline='') as source:
        rows = csv.DictReader(source)
        if not {'t', 'x', 'y', 'status'}.issubset(rows.fieldnames or []):
            raise ValueError('Expected columns t, x, y, status')
        return visits(rows, zones, max_gap)


def write_visits(path, result):
    with path.open('x', encoding='utf-8', newline='') as output:
        csv.writer(output).writerows([COLUMNS, *result])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory', type=Path)
    parser.add_argument('zones', type=Path)
    parser.add_argument('--max-gap', type=float, required=True, metavar='SECONDS')
    parser.add_argument('--out', type=Path, help='Create a new UTF-8 CSV; default is stdout')
    args = parser.parse_args()
    try:
        zones = load_zones(args.zones)
        result = read_visits(args.trajectory, zones, args.max_gap)
        if args.out:
            write_visits(args.out, result)
        else:
            csv.writer(sys.stdout, lineterminator='\n').writerows([COLUMNS, *result])
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
