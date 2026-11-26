"""Motion compensation of one lidar sweep using odometry poses.

Each point carries ``time`` in seconds after the message stamp. The pose of
base_link at the point time is interpolated from odometry and every point is
re-expressed in base_link at the reference time (the message stamp by default).
"""
import numpy as np

from . import se3


def deskew(pts_base, point_times, stamp, odom, bins=16, ref_offset=0.0, max_extrapolation=0.12):
    """Return points in base_link at ``stamp + ref_offset``.

    ``odom`` is an OdomBuffer of T_odom_base. Raises LookupError if odometry
    does not cover the sweep: the caller decides how to report that.
    """
    if len(pts_base) == 0:
        return pts_base
    tmin, tmax = float(point_times.min()), float(point_times.max())
    ref = odom.at(stamp + ref_offset, max_extrapolation)
    if ref is None:
        raise LookupError('no odometry at reference time %.6f' % (stamp + ref_offset))
    if tmax - tmin < 1e-6:
        return pts_base
    edges = np.linspace(tmin, tmax, bins + 1)
    idx = np.clip(np.searchsorted(edges, point_times, side='right') - 1, 0, bins - 1)
    centers = (edges[:-1] + edges[1:]) / 2
    ref_inv = se3.inv(ref)
    out = np.empty_like(pts_base)
    for b in range(bins):
        sel = idx == b
        if not np.any(sel):
            continue
        pose = odom.at(stamp + centers[b], max_extrapolation)
        if pose is None:
            raise LookupError('no odometry at point time %.6f' % (stamp + centers[b]))
        out[sel] = se3.apply(ref_inv @ pose, pts_base[sel])
    return out
