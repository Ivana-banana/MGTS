"""Experimental local point-to-line registration diagnostics for a planar scan."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def diagnose(xy, *, radius_m=.35, neighbors=12, max_variance_ratio=.08):
    """Measure geometric constraints at a correct alignment, in the scan frame.

    The centered Jacobian uses [dx, dy, lever_arm * dyaw], all in metres.
    Condition numbers of the resulting Gram matrices are dimensionless.
    """
    points = np.asarray(xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Expected finite lidar endpoints of shape (N, 2)')
    if not np.isfinite(radius_m) or radius_m <= 0 or neighbors < 4 or not 0 <= max_variance_ratio < 1:
        raise ValueError('Invalid neighborhood parameters')
    points = np.unique(points, axis=0)
    if len(points) < 4:
        raise ValueError('Too few distinct scan points')
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=min(neighbors, len(points)),
                                    distance_upper_bound=radius_m)
    normals, supports = [], []
    for point, dist, idx in zip(points, distances, indices):
        patch = points[idx[np.isfinite(dist)]]
        if len(patch) < 4:
            continue
        centered = patch - patch.mean(axis=0)
        values, axes = np.linalg.eigh(centered.T @ centered)
        if values[1] <= 1e-12 or values[0] > max_variance_ratio * values[1]:
            continue
        supports.append(point)
        normals.append(axes[:, 0])
    if len(normals) < 4:
        raise ValueError('Too few locally linear neighborhoods to estimate normals')
    normals = np.asarray(normals)
    supports = np.asarray(supports)
    center = supports.mean(axis=0)
    offset = supports - center
    lever_arm = np.sqrt(np.mean(np.sum(offset**2, axis=1)))
    yaw_column = (-normals[:, 0]*offset[:, 1] + normals[:, 1]*offset[:, 0]) / lever_arm
    jacobian = np.column_stack((normals, yaw_column))
    gram = jacobian.T @ jacobian / len(normals)
    values = np.maximum(0., np.linalg.eigvalsh(gram))
    translation_values, translation_axes = np.linalg.eigh(gram[:2, :2])
    rank = int(np.count_nonzero(values > 1e-8 * values[-1]))
    return dict(unique_points=len(points), usable_normals=len(normals),
                normal_eigenvalues=np.maximum(0., translation_values).tolist(),
                weakest_translation_xy=translation_axes[:, 0].tolist(),
                pivot_xy_m=center.tolist(), lever_arm_m=float(lever_arm),
                information_eigenvalues=values.tolist(), numerical_rank=rank,
                condition_number=float(values[-1]/values[0]) if rank == 3 else None)


def main():
    parser = argparse.ArgumentParser(prog='python -m mgts_nav.observability', description=__doc__)
    parser.add_argument('--bag', type=Path, help='Use the first processed lidar scan')
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    if args.bag is not None:
        if args.config is None:
            parser.error('--bag requires --config')
        from .bag import load_config, scans
        try:
            first = next(iter(scans(args.bag, load_config(args.config))), None)
        except (OSError, KeyError, ValueError) as exc:
            parser.exit(2, f'{exc}\n')
        if first is None:
            parser.error('Recording contains no lidar scans')
        report = dict(t=first.t, geometry=diagnose(first.xy))
    else:
        along = np.linspace(-3., 3., 121)
        wall = np.column_stack((along, np.ones_like(along)))
        corner_wall = np.column_stack((np.full_like(along, 3.), along + 4.))
        report = dict(parallel_walls=diagnose(np.vstack((wall, wall - [0., 2.]))),
                      perpendicular_walls=diagnose(np.vstack((wall, corner_wall))))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
