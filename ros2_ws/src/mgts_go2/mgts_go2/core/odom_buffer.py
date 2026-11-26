"""Time-indexed buffer of odometry poses with interpolation."""
import bisect

import numpy as np

from . import se3


class OdomBuffer:
    """Stores T_odom_base by header time. Times must arrive in non-decreasing order."""

    def __init__(self, horizon=30.0):
        self.horizon = horizon
        self.times = []
        self.poses = []

    def add(self, stamp, t_odom_base):
        if self.times and stamp < self.times[-1]:
            return False  # out of order: ignore, never reorder history
        if self.times and stamp == self.times[-1]:
            self.poses[-1] = t_odom_base
            return True
        self.times.append(stamp)
        self.poses.append(t_odom_base)
        cut = bisect.bisect_left(self.times, stamp - self.horizon)
        if cut > 0:
            del self.times[:cut]
            del self.poses[:cut]
        return True

    def __len__(self):
        return len(self.times)

    def span(self):
        return (self.times[0], self.times[-1]) if self.times else (None, None)

    def at(self, stamp, max_extrapolation=0.1):
        """Interpolated pose, or None when the stamp is outside the buffer by more than the limit."""
        if not self.times:
            return None
        i = bisect.bisect_left(self.times, stamp)
        if i < len(self.times) and self.times[i] == stamp:
            return self.poses[i]
        if i == 0:
            return self.poses[0] if self.times[0] - stamp <= max_extrapolation else None
        if i == len(self.times):
            return self.poses[-1] if stamp - self.times[-1] <= max_extrapolation else None
        t0, t1 = self.times[i - 1], self.times[i]
        return se3.interpolate(self.poses[i - 1], self.poses[i], (stamp - t0) / (t1 - t0))

    def many(self, stamps, max_extrapolation=0.1):
        """Interpolate several stamps; raises if any is unavailable."""
        out = []
        for s in stamps:
            p = self.at(float(s), max_extrapolation)
            if p is None:
                raise LookupError('odometry not available at %.6f (buffer %s)' % (s, self.span()))
            out.append(p)
        return np.array(out)
