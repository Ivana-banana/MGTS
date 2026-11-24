"""Offline roll/pitch from body-frame angular rates and positive-up specific force."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


GRAVITY = 9.80665


@dataclass
class Leveler:
    times: np.ndarray
    up_in_base: np.ndarray
    max_gap_s: float

    def at(self, t):
        """Return base -> level at t, with zero yaw in ZYX attitude convention."""
        if not np.isfinite(t) or t < self.times[0] or t > self.times[-1]:
            raise ValueError("No IMU coverage at requested time")
        hi = self.times.searchsorted(t)
        if self.times[hi] == t:
            up = self.up_in_base[hi]
        else:
            lo = hi - 1
            dt = self.times[hi] - self.times[lo]
            if dt > self.max_gap_s:
                raise ValueError("IMU gap at requested time")
            fraction = (t - self.times[lo]) / dt
            up = (1 - fraction) * self.up_in_base[lo] + fraction * self.up_in_base[hi]
        if not np.isfinite(up).all() or np.linalg.norm(up) < 1e-8:
            raise ValueError("Gravity direction unavailable after IMU gap")
        roll = np.arctan2(up[1], up[2])
        pitch = np.arctan2(-up[0], np.hypot(up[1], up[2]))
        return Rotation.from_euler("xyz", [roll, pitch, 0.0])


def static_reference(gyro, acc):
    """Median-based startup. Reject motion, vibration and inconsistent units."""
    bias = np.median(gyro, axis=0)
    gravity = np.median(acc, axis=0)
    g_norm = np.linalg.norm(gravity)
    if not (0.90 * GRAVITY < g_norm < 1.10 * GRAVITY):
        return None
    if np.linalg.norm(bias) > 0.12:
        return None
    if np.quantile(np.linalg.norm(gyro - bias, axis=1), 0.9) > 0.08:
        return None
    if np.quantile(np.linalg.norm(acc - gravity, axis=1), 0.9) > 0.70:
        return None
    return bias, gravity / g_norm, g_norm


def build_leveler(times, angular_velocity_xyz, acceleration_xyz, stationary_s, max_gap_s, *, available=None):
    """Build a complementary gravity filter for an offline ROS IMU sequence.

    All readings are in base: rad/s and m/s². stationary_s seconds at the start
    must be at rest. After a clock gap a fresh stationary interval is required
    to recover tilt. Reference yaw cannot be estimated from gravity.
    """
    times = np.asarray(times, dtype=float)
    gyro = np.asarray(angular_velocity_xyz, dtype=float)
    acc = np.asarray(acceleration_xyz, dtype=float)
    if times.ndim != 1 or len(times) < 2 or gyro.shape != (len(times), 3) or acc.shape != gyro.shape:
        raise ValueError("Expected timestamps and Nx3 arrays of IMU samples")
    if not np.isfinite(times).all() or not np.isfinite(gyro).all() or not np.isfinite(acc).all():
        raise ValueError("Non-finite IMU readings")
    if not np.isfinite([stationary_s, max_gap_s]).all() or min(stationary_s, max_gap_s) <= 0:
        raise ValueError("Stationary duration and maximum gap must be positive")
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("IMU times must increase strictly")
    valid = np.ones(len(times), dtype=bool) if available is None else np.asarray(available, dtype=bool)
    if valid.shape != times.shape:
        raise ValueError("Expected one availability flag per IMU sample")
    bounds = np.r_[0, np.flatnonzero((dt > max_gap_s) | (valid[:-1] != valid[1:])) + 1, len(times)]
    up = np.full((len(times), 3), np.nan)

    for start, stop in zip(bounds, bounds[1:]):
        if not valid[start]:
            continue
        # Search again after a gap, but never bootstrap the first segment from
        # a later moving interval if the configured stationary start is invalid.
        for i in range(start, stop):
            last = times.searchsorted(times[i] + stationary_s)
            if last >= stop:
                break
            reference = static_reference(gyro[i:last + 1], acc[i:last + 1])
            if reference is not None:
                bias, gravity, g_norm = reference
                up[i:last + 1] = gravity
                break
            if start == 0:
                raise ValueError("Unreliable initial stationary IMU interval")
        else:
            continue
        if last >= stop:
            if start == 0 and (stop == len(times) or valid[stop]):
                raise ValueError("Insufficient continuous stationary IMU coverage")
            continue

        for k in range(last + 1, stop):
            step = times[k] - times[k - 1]
            omega = 0.5 * (gyro[k - 1] + gyro[k]) - bias
            predicted = Rotation.from_rotvec(-step * omega).apply(up[k - 1])
            norm = np.linalg.norm(acc[k])
            if abs(norm / g_norm - 1) < 0.08:
                measured = acc[k] / norm
                if predicted @ measured > np.cos(np.deg2rad(12.0)):
                    # Gyro follows fast gait; acceleration corrects slow tilt drift.
                    gain = -np.expm1(-step / 0.8)
                    predicted += gain * (measured - predicted)
            up[k] = predicted / np.linalg.norm(predicted)
    return Leveler(times.copy(), up, max_gap_s)
