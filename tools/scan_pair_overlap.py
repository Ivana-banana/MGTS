"""Replay the predeclared synthetic scan-pair controls for issue 117."""
from itertools import product
import json

import numpy as np
from scipy.spatial import cKDTree

from mgts_nav.metrics import wrap
from mgts_nav.slam import world
from tools.intel_longitudinal import alignment
from tools.intel_scan_pair import search


POLYGONS = [
    [(-3, -3), (6, -3), (7, 1), (5, 5), (0, 6), (-3, 3)],
    [(2, 1), (3, 1), (3, 2), (2, 2)],
    [(-1, -2), (0, -2), (0, -1), (-1, -1)],
]


def cross(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def scan(pose, sector):
    angles = np.deg2rad(np.arange(*sector))
    local = np.column_stack((np.cos(angles), np.sin(angles)))
    rays = np.column_stack((np.cos(angles + pose[2]), np.sin(angles + pose[2])))
    distances = np.full(len(angles), np.inf)
    for polygon in POLYGONS:
        for start, end in zip(polygon, polygon[1:] + polygon[:1]):
            edge = np.subtract(end, start)
            offset = np.subtract(start, pose[:2])
            denom = cross(rays, edge)
            valid = np.abs(denom) > 1e-12
            denom = np.where(valid, denom, 1.)
            distance = cross(offset, edge) / denom
            fraction = cross(offset, rays) / denom
            hit = valid & (distance > 0) & (fraction >= 0) & (fraction <= 1)
            distances = np.minimum(distances, np.where(hit, distance, np.inf))
    valid = distances <= 20
    return local[valid] * distances[valid, None]


def main():
    first = scan(np.zeros(3), (-90, 90))
    tree = cKDTree(first)
    for x, angle, sector in product((.5, 2.5), (5, 45), ((-90, 90), (-45, 45), (45, 105))):
        truth = np.array([x, .2, np.deg2rad(angle)])
        current = scan(truth, sector)
        result = search(first, current)
        placed = world(current, truth)
        overlap = ((tree.query(placed)[0] <= .15).mean() +
                   (cKDTree(placed).query(first)[0] <= .15).mean()) / 2
        true_score, _ = alignment(first, tree, current, truth)
        delta = np.asarray(result["pose"]) - truth
        xy = np.linalg.norm(delta[:2])
        yaw = np.rad2deg(abs(wrap(delta[2])))
        outcome = "refused" if result["refusals"] else ("correct" if xy <= .25 and yaw <= 10 else "false")
        print(json.dumps(dict(x=x, yaw_deg=angle, sector=sector, overlap=overlap, true_score=true_score,
                              error_xy=xy, error_deg=yaw, outcome=outcome, **result)), flush=True)


if __name__ == "__main__":
    main()
