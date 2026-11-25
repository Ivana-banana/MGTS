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

class DriftRecovery:
    """Keep local tracking alive while independently checking a global candidate."""

    def __init__(self, pixels, meta):
        self.pixels, self.meta = pixels, meta
        self.previous = self.previous_pose = self.pending = None
        self.uncertain = False
        self.next_attempt = -np.inf
        self.stats = dict(attempts=0, recoveries=0, history=[])

    def update(self, scan, pose, support, matched, tree):
        from .drift_warning import predict as warning

        alert = matched and warning(scan, self.previous, pose, self.previous_pose,
                                    support, tree, 'recorded')[1]
        self.previous, self.previous_pose = scan, pose.copy()
        if alert and not self.uncertain:
            self.stats['history'].append(dict(t=scan.t, action='warning', pose=pose.tolist()))
            self.uncertain = True
        if not self.uncertain or scan.t < self.next_attempt:
            return None
        pending, self.pending = self.pending, None
        if pending and not 0 < scan.t - pending[1] <= .35:
            self.stats['history'].append(dict(t=scan.t, action='expired'))
            pending = None
        if pending is None:
            self.stats['attempts'] += 1
        try:
            alternative = estimate_initial(scan.xy, self.pixels, self.meta)
        except ValueError as exc:
            self.stats['history'].append(dict(t=scan.t, action='refused', reason=str(exc)))
            self.next_attempt = scan.t + .5
            return None
        if pending is None:
            self.pending = alternative, scan.t, scan.odom
            self.stats['history'].append(dict(t=scan.t, action='proposed', pose=alternative.tolist()))
            return None
        expected = predict(pending[0], pending[2], scan)
        error = relative(expected, alternative)
        if np.linalg.norm(error[:2]) > .20 or abs(error[2]) > np.deg2rad(6):
            self.stats['history'].append(dict(t=scan.t, action='inconsistent'))
            self.next_attempt = scan.t + .5
            return None
        self.stats['history'].append(dict(t=scan.t, action='confirmed', pose=alternative.tolist()))
        self.stats['recoveries'] += 1
        self.uncertain = False
        self.previous_pose = alternative.copy()
        return alternative
