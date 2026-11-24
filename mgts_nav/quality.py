"""Describe scan-to-map support and interruptions on the observed inspection clock."""
from itertools import groupby
from statistics import median

import numpy as np


def inspection_quality(estimates, gap_threshold_s=None):
    """Summarize the scan-rate output of track(); no reference poses are used.

    gap_threshold_s overrides the default 3x median scan interval, which is
    unknown below three scans and unreliable on a few of them.
    """
    if gap_threshold_s is not None:
        gap_threshold_s = float(gap_threshold_s)
        if not np.isfinite(gap_threshold_s) or gap_threshold_s <= 0:
            raise ValueError('Gap threshold must be a positive number of seconds')
    times = np.asarray([row.t for row in estimates], dtype=float)
    if not len(times) or not np.isfinite(times).all():
        raise ValueError('Inspection needs finite scan timestamps')
    intervals = np.diff(times)
    if np.any(intervals <= 0):
        raise ValueError('Inspection scan timestamps must increase')

    # Gaps have no rows. Report them independently of consecutive rejected scans.
    if gap_threshold_s is not None:
        gap_threshold = gap_threshold_s
    else:
        gap_threshold = 3 * float(np.median(intervals)) if len(intervals) >= 2 else None
    gaps = []
    if gap_threshold is not None:
        for i in np.flatnonzero(intervals > gap_threshold):
            gaps.append(dict(start_s=float(times[i]), end_s=float(times[i+1]),
                             duration_s=float(intervals[i])))

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
