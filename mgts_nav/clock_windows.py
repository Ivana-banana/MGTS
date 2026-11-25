"""Diagnose local signed header offsets using independent angular-rate records.

Input CSVs contain t (header seconds) and rate (body yaw rate in rad/s).
Positive lag means sensor headers are later. No input record is rewritten.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def interpolate(t, v, query):
    right = np.searchsorted(t, query)
    hi = np.clip(right, 1, len(t)-1)
    result = np.interp(query, t, v, left=np.nan, right=np.nan)
    hole = (t[hi]-t[hi-1] > .16) & (query != t[hi]) & (query != t[hi-1])
    return np.where(hole, np.nan, result)


def window(start, width, reference, sensor):
    grid = start+np.arange(.05, width, .05)
    target = interpolate(*reference, grid)
    lags = np.arange(-100, 101)*.005
    scores = []
    coverage_available = False
    for lag in lags:
        other = interpolate(*sensor, grid+lag)
        valid = np.isfinite(target) & np.isfinite(other)
        if valid.mean() >= .80:
            coverage_available = True
        if valid.mean() < .80 or min(np.std(target[valid]), np.std(other[valid])) < .035:
            scores.append(-2.)
        else:
            scores.append(np.corrcoef(target[valid], other[valid])[0, 1])
    scores = np.asarray(scores)
    best = int(scores.argmax())
    row = dict(start_s=float(start), end_s=float(start+width), lag_s=None)
    if scores[best] == -2:
        return dict(row, status="low_excitation" if coverage_available else "insufficient_coverage")
    if best in (0, len(lags)-1) or scores[best] < .8:
        return dict(row, status="weak_or_out_of_range")
    if scores[best]-max(scores[abs(lags-lags[best]) > .12]) < .02:
        return dict(row, status="ambiguous")
    return dict(row, status="usable", lag_s=float(lags[best]), correlation=float(scores[best]))


def diagnose(reference, sensor, width=5.):
    if not np.isfinite(width) or width <= .5:
        raise ValueError("Window width must exceed 0.5 seconds")
    for t, rate in (reference, sensor):
        if len(t) != len(rate) or len(t) < 2 or not np.isfinite(np.r_[t,rate]).all():
            raise ValueError("Expected finite paired rate samples")
        if np.any(np.diff(t) <= 0):
            raise ValueError("Header reset or duplicate: split at the non-increasing timestamp before fitting")
    count = int((reference[0][-1]-reference[0][0]+1e-6)//width)
    windows = [window(reference[0][0]+i*width, width, reference, sensor) for i in range(count)]
    segments, jumps, group = [], [], []
    def finish():
        if not group:
            return
        centers = np.array([(r["start_s"]+r["end_s"])/2 for r in group])
        lags = np.array([r["lag_s"] for r in group])
        slope = float(np.polyfit(centers-centers[0], lags, 1)[0]) if len(group) >= 3 else None
        kind = "short" if slope is None else ("drift" if abs(slope) >= .001 else "constant")
        segments.append(dict(start_s=group[0]["start_s"], end_s=group[-1]["end_s"], type=kind,
                             slope_s_per_s=slope, median_lag_s=float(np.median(lags))))
        group.clear()
    for row in windows:
        if row["lag_s"] is None:
            finish()
            continue
        if group and abs(row["lag_s"]-group[-1]["lag_s"]) > .08:
            jumps.append(row["start_s"])
            finish()
        group.append(row)
    finish()
    return dict(windows=windows, segments=segments, jumps_s=jumps)


def read_rates(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        values = [(float(r["t"]), float(r["rate"])) for r in csv.DictReader(stream)]
    return np.asarray(values).reshape(-1,2).T


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("sensor", type=Path)
    parser.add_argument("--window-s", type=float, default=5)
    args = parser.parse_args()
    print(json.dumps(diagnose(read_rates(args.reference), read_rates(args.sensor), args.window_s), indent=2))

