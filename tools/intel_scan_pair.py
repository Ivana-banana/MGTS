"""Bounded three-dimensional pose search on the predeclared Intel laser pairs."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from mgts_nav.metrics import read_csv, relative_pose, wrap
from mgts_nav.slam import rotation
from tools.intel_lab import laser_rows, points
from tools.intel_longitudinal import alignment


TIMES = (976053533.696805, 976053540.953401, 976053544.695967, 976053548.341444)


def grid(first, current, trees, xs, ys, degrees):
    xy = np.array([(x, y) for x in xs for y in ys if abs(x) <= 4.000001 and abs(y) <= 4.000001])
    rows = []
    for angle in degrees:
        yaw = np.deg2rad((angle + 180) % 360 - 180)
        rot = rotation(yaw)
        for batch in np.array_split(xy, max(1, (len(xy) + 63) // 64)):
            forward = trees[0].query(current @ rot.T + batch[:, None, :], distance_upper_bound=.25)[0]
            backward = trees[1].query((first - batch[:, None, :]) @ rot, distance_upper_bound=.25)[0]
            scores = (np.minimum(forward, .25).mean(axis=1) + np.minimum(backward, .25).mean(axis=1)) / 2
            rows.extend((s, x, y, yaw) for s, (x, y) in zip(scores, batch))
    return rows


def separate(a, b):
    return np.hypot(a[1]-b[1], a[2]-b[2]) > .5 or abs(wrap(a[3]-b[3])) > np.deg2rad(12)


def search(first, current):
    trees = cKDTree(first), cKDTree(current)
    axis = np.arange(-10, 11) * .4
    evaluated = grid(first, current, trees, axis, axis, range(-180, 180, 15))
    seeds = []
    for row in sorted(evaluated):
        if all(separate(row, s) for s in seeds):
            seeds.append(row)
            if len(seeds) == 16:
                break
    for seed in seeds:
        for step, angle_step, angle_count in ((.1, 3, 5), (.025, 1, 3)):
            _, x, y, yaw = seed
            delta = np.arange(-4, 5) * step
            angles = np.rad2deg(yaw) + np.arange(-angle_count, angle_count+1)*angle_step
            candidates = grid(first, current, trees, x+delta, y+delta, angles)
            evaluated.extend(candidates)
            seed = min(candidates)
    ordered = sorted(evaluated)
    best = ordered[0]
    rival = next(r for r in ordered if separate(r, best))
    score, support = alignment(first, trees[0], current, best[1:])
    reasons = []
    if score > .18:
        reasons.append("score")
    if support < 20:
        reasons.append("support")
    if max(abs(best[1]), abs(best[2])) >= 4 - 1e-8:
        reasons.append("boundary")
    if rival[0] <= best[0] + .01:
        reasons.append("ambiguous")
    return dict(pose=best[1:], score=score, mutual=support, rival_pose=rival[1:], rival_score=rival[0],
                refusals=reasons, evaluations=len(evaluated))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "reference", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    clouds = {}
    for r in laser_rows(args.source):
        stamp = float(r[-3])
        if stamp in TIMES:
            if stamp in clouds:
                raise ValueError(f"Duplicated scan {stamp}")
            clouds[stamp] = points(np.asarray(r[2:182], float))[:, :2].astype(float)
    if set(clouds) != set(TIMES):
        raise ValueError("Missing a fixed inspection scan")
    reports = [dict(t=t, **search(clouds[TIMES[0]], clouds[t])) for t in TIMES[1:]]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"blind.json").write_text(json.dumps(reports, indent=2, allow_nan=False))
    refs = read_csv(args.reference)
    if tuple(r.t for r in refs) != TIMES:
        raise ValueError("Unexpected external reference times")
    for row, ref in zip(reports, refs[1:]):
        delta = np.asarray(row["pose"]) - relative_pose(refs[0], ref)
        row.update(candidate_error_xy=float(np.linalg.norm(delta[:2])),
                   candidate_error_deg=float(np.rad2deg(abs(wrap(delta[2])))))
    print(json.dumps(reports, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
