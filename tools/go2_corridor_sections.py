"""Compare corridor cross-sections observed on the outward and return Go2 mapping legs."""
import argparse
import json
from pathlib import Path

import numpy as np

from mgts_nav.localize3d import scans


def profile_section(points, y):
    slab = points[(abs(points[:, 1] - y) <= .20)
                  & (points[:, 2] >= .65) & (points[:, 2] <= 1.50)]
    sides = []
    for lo, hi in ((-.2, 2.5), (3.4, 5.4)):
        x = slab[(slab[:, 0] >= lo) & (slab[:, 0] <= hi), 0]
        if len(x) < 10:
            sides.append({'x_m': None, 'support': 0, 'returns': len(x)})
            continue
        counts, edges = np.histogram(x, bins=np.arange(lo, hi + .06, .06))
        peak = int(counts.argmax())
        center = (edges[peak] + edges[peak + 1]) / 2
        core = x[abs(x - center) <= .12]
        sides.append({"x_m": float(np.median(core)), "support": len(core), "returns": len(x)})
    left, right = sides
    width = None if left['x_m'] is None or right['x_m'] is None else right['x_m'] - left['x_m']
    return {'left': left, 'right': right, 'width_m': width}


def measure(bag):
    legs = {"outward": [], "return": []}
    first = None
    for t, points, pose, _ in scans(bag):
        if first is None:
            first = t
        elapsed = t - first
        if elapsed <= 35:
            legs["outward"].append(pose.apply(points))
        if elapsed >= 94:
            legs["return"].append(pose.apply(points))
    tables = {}
    for leg, observations in legs.items():
        cloud = np.concatenate(observations)
        tables[leg] = {y: profile_section(cloud, y) for y in (3, 7, 10)}
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mapping", type=Path)
    print(json.dumps(measure(parser.parse_args().mapping), indent=2))
