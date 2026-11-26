"""Synthetic room and sweeps for tests: floor, walls and a few boxes, robot base 0.31 m above the floor."""
import numpy as np

from mgts_go2.core import se3

BASE_HEIGHT = 0.31


def room(spacing=0.05, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    xs = np.arange(0, 10 + 1e-9, spacing)
    ys = np.arange(0, 8 + 1e-9, spacing)
    zs = np.arange(0, 2 + 1e-9, spacing)
    gx, gy = np.meshgrid(xs, ys)
    pts.append(np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)]))           # floor
    for x0 in (0.0, 10.0):
        gy2, gz = np.meshgrid(ys, zs)
        pts.append(np.column_stack([np.full(gy2.size, x0), gy2.ravel(), gz.ravel()]))   # walls x
    for y0 in (0.0, 8.0):
        gx2, gz = np.meshgrid(xs, zs)
        pts.append(np.column_stack([gx2.ravel(), np.full(gx2.size, y0), gz.ravel()]))   # walls y
    # asymmetric boxes so that x, y and yaw are all observable
    for (cx, cy, sx, sy, h) in [(2.0, 2.0, 0.8, 0.5, 1.0), (7.5, 5.5, 0.6, 1.2, 0.8), (4.0, 6.5, 1.5, 0.4, 1.2)]:
        bx = np.arange(cx - sx / 2, cx + sx / 2 + 1e-9, spacing)
        by = np.arange(cy - sy / 2, cy + sy / 2 + 1e-9, spacing)
        bz = np.arange(0, h + 1e-9, spacing)
        for x0 in (cx - sx / 2, cx + sx / 2):
            g1, g2 = np.meshgrid(by, bz)
            pts.append(np.column_stack([np.full(g1.size, x0), g1.ravel(), g2.ravel()]))
        for y0 in (cy - sy / 2, cy + sy / 2):
            g1, g2 = np.meshgrid(bx, bz)
            pts.append(np.column_stack([g1.ravel(), np.full(g1.size, y0), g2.ravel()]))
    p = np.vstack(pts)
    return p + rng.normal(0, 0.003, p.shape)


def corridor(length=30.0, width=2.0, spacing=0.05):
    """Featureless corridor along x: floor and two walls (x is unobservable for a lidar inside)."""
    xs = np.arange(0, length + 1e-9, spacing)
    ys = np.arange(0, width + 1e-9, spacing)
    zs = np.arange(0, 2 + 1e-9, spacing)
    gx, gy = np.meshgrid(xs, ys)
    pts = [np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])]
    for y0 in (0.0, width):
        g1, g2 = np.meshgrid(xs, zs)
        pts.append(np.column_stack([g1.ravel(), np.full(g1.size, y0), g2.ravel()]))
    return np.vstack(pts)


def base_pose(x, y, yaw):
    return se3.make(se3.rot_z(yaw), (x, y, BASE_HEIGHT))


def sweep(world, t_world_base, max_range=5.0, n=900, seed=0, noise=0.01):
    """Points of ``world`` within range, expressed in base_link, randomly thinned to ``n``."""
    rng = np.random.default_rng(seed)
    local = se3.apply(se3.inv(t_world_base), world)
    r = np.linalg.norm(local, axis=1)
    sel = local[(r > 0.3) & (r < max_range)]
    if len(sel) > n:
        sel = sel[rng.choice(len(sel), n, replace=False)]
    return sel + rng.normal(0, noise, sel.shape)
