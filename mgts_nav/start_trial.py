"""Compare unknown-start strategies before scoring against a held-out trajectory."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import initialization, multiscan
from .bag import load_config, scans
from .coarse_start import estimate
from .grid import read_map
from .metrics import Pose, evaluate, read_csv, wrap
from .slam import track, world


def baseline(window, pixels, meta, stats):
    # Count completed coarse sweeps by computing the same known-free tile count.
    # Wrap individual ICP / final optimizations to count actual calls.
    stride = max(1, round(.30 / meta["resolution"]))
    free = 1 - np.flipud(pixels).astype(float)/255 < meta["free_thresh"]
    tile_count = len(np.unique(np.argwhere(free)//stride, axis=0))
    stats.update(coarse=0, icp=0, powell=0)
    with patch.object(initialization, "match", wraps=initialization.match) as icp, \
         patch.object(initialization, "minimize", wraps=initialization.minimize) as opt, \
         patch.object(initialization, "rotation", wraps=initialization.rotation) as yaw_sweeps:
        try:
            try:
                return initialization.estimate_initial(window[0].xy, pixels, meta)
            except ValueError as exc:
                stats["single_refusal"] = str(exc)
                return multiscan.estimate_multiscan(window[:multiscan.MAX_SCANS], pixels, meta)
        finally:
            stats.update(coarse=tile_count*yaw_sweeps.call_count, icp=icp.call_count, powell=opt.call_count)


def compare(bag, config, map_path, reference):
    window = list(scans(bag, load_config(config)))
    if not window:
        raise ValueError("Empty inspection")
    pixels, meta, points = read_map(map_path)
    methods = {"baseline": lambda stats: baseline(window, pixels, meta, stats),
               "coarse_to_fine": lambda stats: estimate(window[0], pixels, meta, stats)}
    report = {}
    starts = {}
    for name, search in methods.items():
        stats = {}
        tick = perf_counter()
        try:
            starts[name] = search(stats)
        except ValueError as exc:
            stats.update(outcome="refused", reason=str(exc))
        stats["seconds"] = perf_counter() - tick
        report[name] = stats

    # Load the reference only after both searches have finished.
    truth = read_csv(reference)
    np.testing.assert_allclose([s.t for s in window], [p.t for p in truth], atol=1e-6, rtol=0)
    anchor = np.array(truth[0][1:])
    for name, start in starts.items():
        rows = list(track(iter(window), start, map_points=points))
        xy_error = np.linalg.norm(start[:2]-anchor[:2])
        angle = abs(wrap(start[2]-anchor[2]))
        report[name].update(outcome="correct" if xy_error <= .25 and angle <= np.deg2rad(10) else "false_accept",
                            start_error_m=xy_error, start_error_deg=np.rad2deg(angle),
                            matched=sum(row.status == "matched" for row in rows),
                            metrics=evaluate([Pose(row.t, *row.pose) for row in rows], truth))
    xy = window[0].xy
    xy = xy[np.linalg.norm(xy-window[0].origin, axis=1) <= 35]
    distance = cKDTree(points).query(world(xy, anchor))[0]
    audit = dict(reference_pose_score=float(np.exp(-.5*(distance/max(.1, meta["resolution"]))**2).mean()),
                 reference_fraction_within_16cm=float((distance < .16).mean()))
    best = report["coarse_to_fine"].get("best_refined_pose")
    if best is not None:
        audit.update(candidate_error_m=np.linalg.norm(best[:2]-anchor[:2]),
                     candidate_error_deg=np.rad2deg(abs(wrap(best[2]-anchor[2]))))
    report["post_search_audit"] = audit
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bag", "profile", "map", "reference"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.bag, args.profile, args.map, args.reference), indent=2, allow_nan=False))
