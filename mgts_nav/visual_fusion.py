"""Use calibrated visual rotations on the corrected LiDAR header clock.

Each UTF-8 JSONL line contains t0 and t1 in seconds plus the raw pair inputs
described in mgts_nav.vision.estimate. Precomputed "accepted"/"yaw_rad" fields
are never trusted. Supply pairs in chronological order without overlaps.
Gaps are allowed; no rotation is assigned to an uncovered part of a scan.
A pair may span a LiDAR boundary: within each accepted pair (at most 0.2 s),
the offline resampling assumes constant yaw rate. Camera rotations add only
over the intersection of that pair with the LiDAR interval.

For a fully covered interval with no gyroscope estimate, use the visual yaw.
When both exist and differ by at most 5 degrees, blend 20% camera and 80% IMU.
A larger difference preserves the gyro estimate. These conservative defaults
are experimental parameters, to be checked against the actual sensor cadence.
The first scan has no preceding interval and contributes to no fusion counter.
"""
import json
import math
from bisect import bisect_right
from dataclasses import dataclass, replace
from pathlib import Path

from .vision import estimate


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    yaw: float | None


def _read_pairs(path, stats):
    pairs = []
    with Path(path).open(encoding='utf-8-sig') as source:
        for line_num, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                start, end = float(row['t0']), float(row['t1'])
                if not math.isfinite(start) or not math.isfinite(end) or start >= end:
                    raise ValueError('Expected finite t0 < t1 in seconds')
                if pairs and start < pairs[-1].end:
                    raise ValueError('Overlapping or out-of-order camera intervals')
                yaw = None
                if end - start > .2 + 2 * max(math.ulp(start), math.ulp(end)):
                    stats['long_pairs'] += 1
                else:
                    result = estimate(row)
                    if result['accepted']:
                        yaw = result['yaw_rad']
                        stats['accepted_pairs'] += 1
                    else:
                        stats['geometry_rejected_pairs'] += 1
                        reasons = stats['geometry_rejection_reasons']
                        reason = result['reason']
                        reasons[reason] = reasons.get(reason, 0) + 1
                pairs.append(Interval(start, end, yaw))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{path}:{line_num}: {exc}') from exc
    return pairs


def _interval_yaw(pairs, ends, start, end):
    index = bisect_right(ends, start)
    cursor, total = start, 0.
    while cursor < end and index < len(pairs):
        pair = pairs[index]
        if pair.start > cursor or pair.yaw is None:
            return None
        until = min(pair.end, end)
        total += pair.yaw * ((until - cursor) / (pair.end - pair.start))
        cursor = until
        index += 1
    return total if cursor == end else None


def apply_visual(scans, path, stats):
    """Yield scans; increment caller-provided diagnostics once per LiDAR span.

    fallback + blend + rejected counts the spans, not the input feature pairs.
    uncovered includes absent pairs, time gaps and failed/long pairs; conflict
    counts disagreement with an available gyro. These two sum to rejected.
    An overlap or reversed clock is a malformed file and raises ValueError.
    """
    for key in ('fallback', 'blend', 'rejected', 'uncovered', 'conflict', 'long_pairs',
                'geometry_rejected_pairs', 'accepted_pairs'):
        stats.setdefault(key, 0)
    stats.setdefault('geometry_rejection_reasons', {})
    pairs = _read_pairs(path, stats)
    ends = [pair.end for pair in pairs]
    previous = None
    for scan in scans:
        if not math.isfinite(scan.t) or (previous is not None and scan.t <= previous):
            raise ValueError('LiDAR scan times must be finite and strictly increasing')
        if previous is not None:
            yaw = _interval_yaw(pairs, ends, previous, scan.t)
            if yaw is None:
                stats['uncovered'] += 1
                stats['rejected'] += 1
            elif scan.gyro_delta is None:
                scan = replace(scan, gyro_delta=yaw)
                stats['fallback'] += 1
            elif abs(yaw - scan.gyro_delta) <= math.radians(5):
                scan = replace(scan, gyro_delta=.8 * scan.gyro_delta + .2 * yaw)
                stats['blend'] += 1
            else:
                stats['conflict'] += 1
                stats['rejected'] += 1
        previous = scan.t
        yield scan
