"""Describe scan-to-map support and interruptions on the observed inspection clock."""
from itertools import groupby
from statistics import median

import numpy as np


def scan_intervals(estimates):
    """Return successive intervals, the gap threshold and indices following gaps."""
    times = np.asarray([row.t for row in estimates], dtype=float)
    if not len(times) or not np.isfinite(times).all():
        raise ValueError('Inspection needs finite scan timestamps')
    intervals = np.diff(times)
    if np.any(intervals <= 0):
        raise ValueError('Inspection scan timestamps must increase')

    # Gaps have no rows. Report them independently of consecutive rejected scans.
    gap_threshold = 3 * float(np.median(intervals)) if len(intervals) >= 2 else None
    ends = np.flatnonzero(intervals > gap_threshold) + 1 if gap_threshold is not None else []
    return intervals, gap_threshold, ends


def inspection_quality(estimates):
    """Summarize the scan-rate output of track(); no reference poses are used."""
    intervals, gap_threshold, ends = scan_intervals(estimates)
    times = [row.t for row in estimates]
    gaps = [dict(start_s=float(times[i-1]), end_s=float(times[i]),
                 duration_s=float(intervals[i-1])) for i in ends]

    support = []
    losses = []
    for status, rows in groupby(enumerate(estimates), key=lambda item: item[1].status):
        if status == 'matched':
            support.extend(row.inliers for _, row in rows)
        elif status == 'prediction_only':
            indices = [index for index, _ in rows]
            first, last = indices[0], indices[-1]
            losses.append(dict(first_prediction_s=float(times[first]), last_prediction_s=float(times[last]),
                               scans=len(indices),
                               previous_match_s=float(times[first - 1]) if first else None,
                               next_match_s=float(times[last + 1]) if last + 1 < len(times) else None))
        else:
            raise ValueError(f'Unexpected inspection status: {status}')

    return dict(matched_scan_fraction=len(support) / len(times),
                median_inlier_fraction_on_matches=float(median(support)) if support else None,
                max_scan_interval_s=float(intervals.max()) if len(intervals) else None,
                gap_threshold_s=gap_threshold, scan_gaps=gaps, prediction_runs=losses)
