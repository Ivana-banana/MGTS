"""Recover a lost inspection track using two independent global map matches."""
import numpy as np

from .initialization import estimate_initial
from .slam import predict, relative


class Recovery:
    """Propose after three consecutive misses and verify on the next scan."""

    def __init__(self, pixels, meta):
        self.pixels = pixels
        self.meta = meta
        self.misses = 0
        self.pending = None
        self.stats = {'attempts': 0, 'recoveries': 0}

    def update(self, scan, matched):
        if matched:
            self.misses = 0
            self.pending = None
            return None

        if self.pending is None:
            self.misses += 1
            if self.misses < 3:
                return None
            self.misses = 0
            self.stats['attempts'] += 1
            if np.isfinite(scan.t):
                try:
                    candidate = estimate_initial(scan.xy, self.pixels, self.meta)
                except ValueError:
                    return None
                self.pending = candidate, scan.t, scan.odom
            return None

        candidate, timestamp, odom = self.pending
        self.pending = None
        # A repeated timestamp or long outage cannot verify the previous scan.
        if not 0 < scan.t - timestamp <= 1.0:
            return None
        try:
            observed = estimate_initial(scan.xy, self.pixels, self.meta)
        except ValueError:
            return None

        expected = predict(candidate, odom, scan)
        error = relative(expected, observed)
        if np.linalg.norm(error[:2]) <= .20 and abs(error[2]) <= np.deg2rad(6):
            self.stats['recoveries'] += 1
            return observed
        return None
