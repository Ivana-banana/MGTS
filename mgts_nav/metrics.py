"""Evaluate planar trajectories in a shared metric frame."""

import argparse
import csv
import json
import math
from bisect import bisect_left
from pathlib import Path
from statistics import median
from typing import NamedTuple


class Pose(NamedTuple):
    t: float
    x: float
    y: float
    yaw: float


def read_csv(path):
    poses = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if not set(Pose._fields).issubset(rows.fieldnames or []):
            if any(';' in name for name in rows.fieldnames or []):
                raise ValueError(f"{path}: columns are separated by ';'. Save the file as CSV with comma "
                                 "separators and a dot as the decimal mark (t,x,y,yaw)")
            raise ValueError(f"{path}: required columns are t,x,y,yaw")
        for line, row in enumerate(rows, 2):
            try:
                pose = Pose(*(float(row[key]) for key in Pose._fields))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line}: invalid number") from error
            if not all(map(math.isfinite, pose)):
                raise ValueError(f"{path}:{line}: non-finite value")
            if poses and pose.t <= poses[-1].t:
                raise ValueError(f"{path}:{line}: timestamps must increase")
            poses.append(pose)
    if not poses:
        raise ValueError(f"{path}: empty trajectory")
    return poses


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def interpolate(poses, times, t, max_gap):
    right = bisect_left(times, t)
    if right < len(times) and times[right] == t:
        return poses[right]
    if right == 0 or right == len(times):
        return None
    a, b = poses[right - 1], poses[right]
    if b.t - a.t > max_gap:
        return None
    w = (t - a.t) / (b.t - a.t)
    return Pose(t, a.x + w * (b.x - a.x), a.y + w * (b.y - a.y),
                a.yaw + w * wrap(b.yaw - a.yaw))


def relative_pose(a, b):
    dx, dy = b.x - a.x, b.y - a.y
    c, s = math.cos(a.yaw), math.sin(a.yaw)
    return c * dx + s * dy, -s * dx + c * dy, wrap(b.yaw - a.yaw)


def cadence_limit(poses):
    intervals = [b.t - a.t for a, b in zip(poses, poses[1:])]
    return 3 * median(intervals) if intervals else 0.0


def evaluate(estimate, truth, max_gap_s=None):
    if max_gap_s is not None and (not math.isfinite(max_gap_s) or max_gap_s <= 0):
        raise ValueError("max-gap-s must be finite and positive")
    estimate_limit = cadence_limit(estimate) if max_gap_s is None else max_gap_s
    truth_limit = cadence_limit(truth) if max_gap_s is None else max_gap_s
    times = [p.t for p in estimate]
    matched = pairs = 0
    position_sq = translation_sq = rotation_sq = 0.0
    previous = None
    for ref in truth:
        est = interpolate(estimate, times, ref.t, estimate_limit)
        if est is None:
            previous = None
            continue
        matched += 1
        position_sq += (est.x - ref.x) ** 2 + (est.y - ref.y) ** 2
        if previous is not None:
            last_est, last_ref = previous
            # A pair requires a continuous estimator interval even when the
            # truth samples land exactly on opposite sides of an outage.
            left = bisect_left(times, last_ref.t)
            if left and times[left] != last_ref.t:
                left -= 1
            right = bisect_left(times, ref.t)
            continuous = all(times[i+1] - times[i] <= estimate_limit
                             for i in range(left, right))
            if continuous and ref.t - last_ref.t <= truth_limit:
                ex, ey, ea = relative_pose(last_est, est)
                tx, ty, ta = relative_pose(last_ref, ref)
                translation_sq += (ex - tx) ** 2 + (ey - ty) ** 2
                rotation_sq += wrap(ea - ta) ** 2
                pairs += 1
        previous = est, ref
    return {
        "truth_samples": len(truth), "matched_samples": matched,
        "coverage": matched / len(truth), "relative_pairs": pairs,
        "ate_xy_rmse_m": math.sqrt(position_sq / matched) if matched else None,
        "rpe_translation_rmse_m": math.sqrt(translation_sq / pairs) if pairs else None,
        "rpe_rotation_rmse_rad": math.sqrt(rotation_sq / pairs) if pairs else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("estimate")
    parser.add_argument("truth")
    parser.add_argument("--max-gap-s", type=float, help="Maximum continuous sample interval in seconds")
    args = parser.parse_args()
    try:
        result = evaluate(read_csv(args.estimate), read_csv(args.truth), args.max_gap_s)
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
