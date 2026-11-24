"""Report repeated lidar returns in previously mapped free space."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .bag import load_config, scans
from .grid import read_map
from .metrics import read_csv
from .slam import world


def matched_poses(path):
    rows = read_csv(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "status" not in reader.fieldnames:
            raise ValueError("Expected localization trajectory with status")
        return {pose.t: np.array(pose[1:]) for pose, row in zip(rows, reader, strict=True)
                if row["status"] == "matched"}


def scan_groups(points, free, meta):
    cell = meta["resolution"]
    origin = np.asarray(meta["origin"][:2])
    cells = np.unique(np.floor((points - origin) / cell).astype(int), axis=0)
    h, w = free.shape
    x, y = cells.T
    cells = cells[(x >= 0) & (x < w) & (y >= 0) & (y < h)]
    x, y = cells.T
    cells = cells[free[h-1-y, x]]
    if len(cells) < 3:
        return []
    centers = origin + (cells + .5) * cell
    tree = cKDTree(centers)
    graph = tree.sparse_distance_matrix(tree, max(.20, 1.5*cell), output_type="coo_matrix")
    count, labels = connected_components(graph, directed=False)
    return [centers[labels == k].mean(axis=0) for k, size in enumerate(np.bincount(labels, minlength=count))
            if size >= 3]


def detect(stream, poses, pixels, meta):
    cell = meta["resolution"]
    margin = max(1, int(np.ceil(.12 / cell)))
    free = binary_erosion((1 - pixels / 255.) < meta["free_thresh"],
                          structure=np.ones((3, 3), bool), iterations=margin)
    tracks = []
    active = []
    used = total = 0
    last_t = -np.inf
    for scan in stream:
        if scan.t <= last_t:
            raise ValueError("Inspection scan timestamps must increase")
        last_t = scan.t
        total += 1
        active = [tr for tr in active if scan.t - tr["last_s"] <= .5]
        if scan.t not in poses:
            continue
        used += 1
        groups = scan_groups(world(scan.xy, poses[scan.t]), free, meta)
        pairs = sorted((float(np.linalg.norm(center - tr["center"])), i, j)
                       for i, center in enumerate(groups) for j, tr in enumerate(active))
        assigned, taken = set(), set()
        for distance, i, j in pairs:
            if distance > .35:
                break
            if i in assigned or j in taken:
                continue
            assigned.add(i)
            taken.add(j)
            tr = active[j]
            tr["observations"] += 1
            tr["last_s"] = scan.t
            tr["center"] += (groups[i] - tr["center"]) / tr["observations"]
            if tr["observations"] == 3:
                tr["confirmed_s"] = scan.t
        for i, center in enumerate(groups):
            if i not in assigned:
                tr = dict(first_s=scan.t, last_s=scan.t, observations=1, center=center)
                active.append(tr)
                tracks.append(tr)
    events = [dict(first_s=tr["first_s"], confirmed_s=tr["confirmed_s"], last_s=tr["last_s"],
                   x_m=float(tr["center"][0]), y_m=float(tr["center"][1]), observations=tr["observations"])
              for tr in tracks if "confirmed_s" in tr]
    return dict(scans=total, matched_pose_scans=used, events=events)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="Inspection recording")
    parser.add_argument("--map", type=Path, required=True, help="Reference map YAML")
    parser.add_argument("--trajectory", type=Path, required=True, help="Localization CSV for this bag")
    parser.add_argument("--config", type=Path, required=True, help="Sensor configuration used for localization")
    parser.add_argument("--out", type=Path, required=True, help="New JSON report")
    args = parser.parse_args(argv)
    pixels, meta, _ = read_map(args.map)
    result = detect(scans(args.bag, load_config(args.config)), matched_poses(args.trajectory), pixels, meta)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
