"""Compare one-second localization increments to kinematic translation and IMU yaw."""
from math import degrees

import numpy as np

from .metrics import wrap
from .slam import relative


def windows(stream, estimates):
    if len(stream) != len(estimates):
        raise ValueError('Scan and trajectory lengths differ')
    times = np.array([s.t for s in stream], dtype=float)
    if (not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or any(abs(s.t - r.t) > 1e-6 for s, r in zip(stream, estimates))):
        raise ValueError('Expected corresponding, increasing scan times')
    for i, t in enumerate(times):
        # Select the closest sample to the left boundary, with at most 50 ms timing error.
        j = int(np.searchsorted(times, t - 1.))
        if j and abs(times[j-1] - (t-1.)) < abs(times[j] - (t-1.)):
            j -= 1
        translation = yaw = None
        if abs(t - times[j] - 1.) <= .05 and np.all(np.diff(times[j:i+1]) <= .2 + 1e-8):
            step = relative(estimates[j].pose, estimates[i].pose)
            if all(s.odom is not None for s in stream[j:i+1]):
                odom = relative(stream[j].odom, stream[i].odom)
                translation = float(np.linalg.norm(step[:2] - odom[:2]))
            increments = [s.gyro_delta for s in stream[j+1:i+1]]
            if all(v is not None for v in increments):
                yaw = abs(degrees(wrap(step[2] - sum(increments))))
        alert = (translation is not None and translation > .15) or (yaw is not None and yaw > 5.)
        yield dict(t=float(t), translation_m=translation, yaw_deg=yaw,
                   alert=alert if translation is not None or yaw is not None else None)
