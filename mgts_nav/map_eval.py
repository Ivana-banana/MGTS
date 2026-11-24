"""Compare an MGTS map in the demo world frame with the analytic static scene."""
import argparse
import json

import numpy as np
from scipy.spatial import cKDTree

from .demo import WALLS
from .grid import read_map


def segment_distances(points, segments):
    """Shortest Euclidean distance to any finite segment, in metres."""
    nearest = np.full(len(points), np.inf)
    for start, end in segments:
        edge = end - start
        w = np.clip((points - start) @ edge / (edge @ edge), 0, 1)
        distance = np.linalg.norm(points - (start + w[:, None] * edge), axis=1)
        nearest = np.minimum(nearest, distance)
    return nearest


def evaluate_map(points, segments, resolution, tolerance):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be finite and positive")
    if not len(points):
        raise ValueError("No occupied cells")
    distances = segment_distances(points, segments)
    false = distances > tolerance
    tree = cKDTree(points)
    total_length = covered_length = 0.0
    for start, end in segments:
        length = np.linalg.norm(end - start)
        count = int(np.ceil(length / (resolution / 2)))
        fractions = (np.arange(count) + 0.5) / count
        samples = start + fractions[:, None] * (end - start)
        covered_length += length * np.mean(tree.query(samples)[0] <= tolerance)
        total_length += length
    # Moving cylinder in the fixture, plus half a pixel diagonal at its edge.
    path = np.array([[[0.5, -0.9], [2.1, -0.9]]])
    in_sweep = segment_distances(points, path) <= 0.2 + resolution / np.sqrt(2)
    return dict(resolution_m=float(resolution), tolerance_m=float(tolerance),
                occupied_cells=len(points), off_wall_cells=int(false.sum()),
                off_wall_fraction=float(false.mean()),
                off_wall_in_person_sweep=int(np.sum(false & in_sweep)),
                off_wall_outside_person_sweep=int(np.sum(false & ~in_sweep)),
                static_wall_length_m=float(total_length),
                static_wall_length_covered_fraction=float(covered_length / total_length))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", help="map.yaml from synthetic mapping, anchored at [0,0,0]")
    parser.add_argument("--tolerance", type=float, default=0.12, help="Distance in metres")
    args = parser.parse_args()
    try:
        _, meta, points = read_map(args.map)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"{args.map}: cannot read map.yaml: {exc}\n")
    print(json.dumps(evaluate_map(points, WALLS, meta["resolution"], args.tolerance), indent=2))
