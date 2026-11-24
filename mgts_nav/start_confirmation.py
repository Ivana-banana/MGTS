"""Experimental confirmation of a first-scan global pose using later returns."""
from itertools import islice

import numpy as np

from .multiscan import MAX_SCANS, WINDOW_S, estimate_multiscan
from .slam import relative, world


def confirm_start(candidate, first, following, pixels, meta):
    """Return original pose or raise ValueError if later observations cannot verify it."""
    candidate = np.asarray(candidate, dtype=float)
    if candidate.shape != (3,) or not np.isfinite(candidate).all():
        raise ValueError("Invalid candidate pose")
    if not np.isfinite(first.t) or np.shape(first.odom) != (3,) or not np.isfinite(first.odom).all():
        raise ValueError("First scan needs a valid timestamp and odometry")
    window = []
    previous_t = first.t
    for scan in islice(following, MAX_SCANS - 1):
        if not np.isfinite(scan.t) or scan.t <= previous_t:
            raise ValueError("Confirmation needs strictly later scan timestamps")
        if scan.t - first.t > WINDOW_S:
            break
        window.append(scan)
        previous_t = scan.t
    if len(window) < 2:
        raise ValueError("Insufficient later scans to confirm start")
    # This search receives neither the candidate nor the original scan.
    observed = estimate_multiscan(window, pixels, meta)
    step = relative(first.odom, window[0].odom)
    expected = np.r_[world(step[:2], candidate), candidate[2] + step[2]]
    error = relative(expected, observed)
    if np.linalg.norm(error[:2]) > .20 or abs(error[2]) > np.deg2rad(6):
        raise ValueError("Later scans disagree with proposed start")
    return candidate.copy()
