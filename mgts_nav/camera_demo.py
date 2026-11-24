"""Render noisy, rectified observations of static markers on the office's east wall."""
import json

import numpy as np


PERIOD_NS = 25_000_000
K = [[420., 0., 320.], [0., 420., 240.], [0., 0., 1.]]
# Optical right/down/forward correspond to body -Y/-Z/+X.
BASE_FROM_CAMERA = [-0.5, 0.5, -0.5, 0.5]
CAMERA_HEIGHT = 1.0


def _visible_markers(points, pose, walls, rng, stripe, noise_px):
    dx, dy = points[:, 0] - pose[0], points[:, 1] - pose[1]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    depth = c * dx + s * dy
    right, down = s * dx - c * dy, CAMERA_HEIGHT - points[:, 2]
    uv = np.column_stack((320 + 420 * right / depth, 240 + 420 * down / depth))
    uv += rng.normal(0., noise_px, uv.shape)
    keep = (depth > 0) & ((uv >= 0) & (uv < [640, 480])).all(axis=1)
    # Treat lidar obstacles as vertical opaque partitions between the lens and the panel.
    for a, b in walls:
        edge = b - a
        det = dx * edge[1] - dy * edge[0]
        ax, ay = a - pose[:2]
        with np.errstate(divide='ignore', invalid='ignore'):
            along_ray = (ax * edge[1] - ay * edge[0]) / det
            along_edge = (ax * dy - ay * dx) / det
        keep &= ~((along_ray > 0) & (along_ray < 1 - 1e-8) & (along_edge >= 0) & (along_edge <= 1))
    if stripe:
        keep &= points[:, 2] == CAMERA_HEIGHT
    keep &= rng.random(len(points)) >= .12
    return uv, keep


def write_camera_pairs(path, seed, pose_at, walls, start_ns, *, noise_px=.15):
    """Only the simulator sees the analytic path. Emit observations on the ROS header clock."""
    rng = np.random.default_rng(seed)
    markers = np.column_stack((np.full(200, 5.2), rng.uniform(-1.9, 4.7, 200),
                               rng.uniform(.25, 2.65, 200)))
    markers[:60, 2] = CAMERA_HEIGHT
    previous_uv = previous_mask = previous_t = None
    with path.open('x', encoding='utf-8') as stream:
        # Four frames per lidar period, with endpoints aligned to its 35 ms phase.
        for dt in range(35_000_000, 9_935_000_001, PERIOD_NS):
            elapsed = dt / 1e9
            sec, ns = divmod(start_ns + dt, 1_000_000_000)
            t = sec + ns * 1e-9
            uv, mask = _visible_markers(markers, pose_at('inspection', elapsed), walls, rng,
                                        stripe=5.1 <= elapsed < 5.35, noise_px=noise_px)
            if previous_mask is not None:
                ids = np.flatnonzero(previous_mask & mask)
                end = uv[ids].copy()
                # False associations still land inside the image.
                bad = rng.choice(len(ids), size=len(ids) // 12, replace=False)
                end[bad] = rng.uniform([0, 0], [640, 480], (len(bad), 2))
                tracks = [[int(i), *p, *q] for i, p, q in zip(ids, previous_uv[ids], end)]
                pair = dict(t0=previous_t, t1=t, K=K, base_from_camera_xyzw=BASE_FROM_CAMERA,
                            camera_motion_bound_m=.5 * PERIOD_NS / 1e9, min_scene_range_m=3.,
                            tracks=tracks)
                stream.write(json.dumps(pair, separators=(',', ':'), allow_nan=False) + '\n')
            previous_uv, previous_mask, previous_t = uv, mask, t
