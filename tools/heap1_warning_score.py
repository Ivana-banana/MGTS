"""Score a fixed inlier warning on registered HEAP-1 inspection windows."""
import argparse
import csv
import json
import math
from pathlib import Path

from mgts_nav.metrics import cadence_limit, interpolate, read_csv


def warning(status, support):
    if status != 'matched' or support is None or not support.strip():
        return None
    fraction = float(support)
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError(f'Invalid inlier_fraction: {support!r}')
    return fraction < .55


def counts():
    return dict(tp=0, fp=0, tn=0, fn=0)


def record(result, alert, large):
    key = ('tp' if large else 'fp') if alert else ('fn' if large else 'tn')
    result[key] += 1


def rates(result, expected):
    positive = result['tp'] + result['fn']
    negative = result['fp'] + result['tn']
    return dict(**result, sensitivity=result['tp']/positive if positive else None,
                false_alarm_rate=result['fp']/negative if negative else None,
                coverage=(positive + negative)/expected)


def score(trajectory, reference, expected=60):
    poses = read_csv(trajectory)
    truth = read_csv(reference)
    times = [p.t for p in truth]
    gap = cadence_limit(truth)
    if len(poses) > expected:
        raise ValueError(f'{trajectory}: more than {expected} scan poses')
    method, baseline, common = counts(), counts(), counts()
    outside = missing_support = missing_truth = matched = 0
    squared = []
    with trajectory.open(encoding='utf-8-sig', newline='') as stream:
        rows = csv.DictReader(stream)
        if not {'status', 'inlier_fraction'}.issubset(rows.fieldnames or []):
            raise ValueError(f'{trajectory}: missing status or inlier_fraction')
        for pose, row in zip(poses, rows, strict=True):
            if row['status'] != 'matched':
                outside += 1
                continue
            matched += 1
            alert = warning(row['status'], row['inlier_fraction'])
            missing_support += alert is None
            target = interpolate(truth, times, pose.t, gap)
            if target is None:
                missing_truth += 1
                continue
            error = math.hypot(pose.x - target.x, pose.y - target.y)
            large = error > .20
            squared.append(error**2)
            record(baseline, False, large)
            if alert is not None:
                record(common, False, large)
                record(method, alert, large)
    return dict(expected=expected, received=len(poses), missing_scans=expected-len(poses),
                matched=matched, other_status=outside, matched_missing_truth=missing_truth,
                matched_missing_support=missing_support, truth_max_gap_s=gap,
                matched_ate_m=math.sqrt(sum(squared)/len(squared)) if squared else None,
                warning=rates(method, expected), baseline=rates(baseline, expected),
                baseline_same_support=rates(common, expected))


def run(index):
    windows = []
    seen = set()
    for entry in json.loads(index.read_text(encoding='utf-8-sig')):
        key = (entry['map'], entry['first'])
        if key in seen:
            raise ValueError(f'Duplicate window {key}')
        seen.add(key)
        result = score(index.parent / entry['trajectory'], index.parent / entry['truth'])
        windows.append(dict(map=key[0], first=key[1], **result))
    if not windows:
        raise ValueError('Empty window index')
    return dict(support_threshold=.55, error_threshold_m=.20, windows=windows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    try:
        payload = json.dumps(run(args.index), indent=2, allow_nan=False)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(payload + '\n')
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'{exc}\n')
