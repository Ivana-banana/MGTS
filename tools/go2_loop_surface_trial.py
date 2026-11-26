"""Test a lateral loop correction on disjoint corridor sections and inspection scans."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from mgts_nav.localize3d import CorrectionFilter, SurfaceMap, scans, voxel_centroids


def pairs(early, late, windows, zrange):
    result = []
    for y in windows:
        for lo, hi in ((-.2, 2.5), (3.4, 5.4)):
            slices = []
            for cloud in (early, late):
                xval, yval, zval = cloud.T
                mask = ((abs(yval - y) <= .2) & (xval > lo) & (xval < hi)
                        & (zval >= zrange[0]) & (zval < zrange[1]))
                slices.append(cloud[mask])
            if min(map(len, slices)) >= 10:
                a, b = slices
                result.append((a, cKDTree(a), b, cKDTree(b)))
    return result


def distance(samples, shift):
    costs = []
    offset = np.array([shift, 0, 0])
    for a, ta, b, tb in samples:
        forward = ta.query(b - offset)[0]
        reverse = tb.query(a + offset)[0]
        costs.append((np.median(forward) + np.median(reverse)) / 2)
    return float(np.mean(costs)) if costs else None


def evaluate(inspection, points):
    surface, filt = SurfaceMap(points), CorrectionFilter()
    scores, room = [], []
    matched = 0
    for t, cloud, prior, _ in scans(inspection):
        test = np.arange(len(cloud)) % 4 == 0
        fit, accepted = surface.align(cloud[~test], prior)
        pose = filt.update(t, prior, fit, accepted)
        score = surface.score(cloud[test], pose)
        scores.append(score)
        matched += int(accepted)
        if prior.xyz[0] < .7 and prior.xyz[1] > 11:
            room.append(score)
    return dict(scans=len(scores), matched=matched, heldout_m=float(np.mean(scores)),
                room_scans=len(room), room_m=float(np.mean(room)) if room else None)


def measure(mapping, inspection, output):
    output.mkdir(parents=True)
    records = [(t, pose.apply(cloud)) for t, cloud, pose, _ in scans(mapping)]
    elapsed = np.array([t for t, _ in records]) - records[0][0]
    clouds = [xyz for _, xyz in records]
    early, late = [voxel_centroids(np.concatenate([c for c, use in zip(clouds, mask) if use]), .06)
                   for mask in (elapsed <= 35, elapsed >= 94)]
    training = pairs(early, late, (0, 1.5, 4.5, 5.5, 8.5, 11.5), (.65, 1.5))
    if not training:
        raise ValueError("No matching training sections")
    shifts = np.linspace(-.75, .75, 61)
    shift = float(min(shifts, key=lambda x: distance(training, x)))
    controls = []
    for y in (3, 7, 10):
        for height in ((.65, 1.), (1., 1.5)):
            groups = pairs(early, late, (y,), height)
            controls.append(dict(y=y, height=height, usable_sides=len(groups),
                                 original_m=distance(groups, 0), corrected_m=distance(groups, shift)))
    u = np.clip((elapsed - 35) / (94 - 35), 0, 1)
    weights = u*u*(3 - 2*u)
    baseline = voxel_centroids(np.concatenate(clouds), .12)
    correction = np.array([shift, 0, 0])
    corrected = voxel_centroids(np.concatenate([c - w*correction for c, w in zip(clouds, weights)]), .12)
    np.savez_compressed(output / "candidate.npz", xyz=corrected)
    return dict(shift_m=shift, train_sides=len(training), train_before_m=distance(training, 0),
                train_after_m=distance(training, shift), controls=controls,
                original=evaluate(inspection, baseline), candidate=evaluate(inspection, corrected))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mapping", "inspection", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    report = json.dumps(measure(**vars(args)), indent=2)
    (args.output / "report.json").write_text(report + "\n", encoding="utf-8")
    print(report)
