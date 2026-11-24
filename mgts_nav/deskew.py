"""Compensate within-scan planar motion using time-stamped kinematic odometry."""
import numpy as np


def deskew_points(xyz_base, offsets_s, scan_t, odom, max_gap_s):
    """Express each return in the base frame at scan_t, preserving height.

    xyz_base[i] is in the base frame at scan_t + offsets_s[i]. Odometry rows are
    (time in seconds, world x, world y, unwrapped world yaw in radians).
    Samples are interpolated offline. The entire interval from the first return
    through the reference pose to the last return must have valid coverage.
    """
    xyz = np.asarray(xyz_base, dtype=float)
    offsets = np.asarray(offsets_s, dtype=float)
    poses = np.asarray(odom, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or offsets.shape != (len(xyz),):
        raise ValueError("Expected N x 3 points and one acquisition offset for each point")
    if not np.all(np.isfinite(offsets)) or not np.isfinite(scan_t):
        raise ValueError("Invalid acquisition time")
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be positive and finite")
    if poses.ndim != 2 or poses.shape[1] != 4 or len(poses) == 0:
        raise ValueError("Expected a non-empty array of odometry rows (t, x, y, yaw)")
    timestamps = poses[:, 0]
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Odometry timestamps must be finite and strictly increasing")

    # Include the target time even if all returns were captured before/after it.
    sample_times = np.r_[scan_t + offsets, scan_t]
    first, last = np.min(sample_times), np.max(sample_times)
    if first < timestamps[0] or last > timestamps[-1]:
        raise ValueError("Odometry does not cover the scan and its reference time")
    lo = timestamps.searchsorted(first, side="right") - 1
    hi = timestamps.searchsorted(last, side="left") + 1
    local = poses[lo:hi]
    if np.any(np.diff(local[:, 0]) > max_gap_s):
        raise ValueError("Odometry gap inside the scan interval")
    if not np.all(np.isfinite(local[:, 1:])):
        raise ValueError("Non-finite odometry inside the scan interval")

    x, y, yaw = [np.interp(sample_times, local[:, 0], v) for v in local[:, 1:].T]
    c, s = np.cos(yaw[-1]), np.sin(yaw[-1])
    dx, dy = x[:-1] - x[-1], y[:-1] - y[-1]
    angle = yaw[:-1] - yaw[-1]
    ca, sa = np.cos(angle), np.sin(angle)
    result = xyz.copy()
    result[:, 0] = c * dx + s * dy + ca * xyz[:, 0] - sa * xyz[:, 1]
    result[:, 1] = -s * dx + c * dy + sa * xyz[:, 0] + ca * xyz[:, 1]
    return result
