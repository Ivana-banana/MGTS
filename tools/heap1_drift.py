"""Attribute accumulated planar position error on the paired HEAP-1 scan clocks."""
import argparse
import json

import numpy as np

from mgts_nav.metrics import read_csv


def rotate(xy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    x, y = xy.T
    return np.column_stack((c*x - s*y, s*x + c*y))


def analyze(estimate, reference):
    est, ref = np.asarray(estimate), np.asarray(reference)
    if est.shape != (300, 4) or ref.shape != est.shape:
        raise ValueError('Expected paired 300-scan HEAP-1 trajectories')
    if not np.allclose(est[:, 0], ref[:, 0], atol=1e-6, rtol=0):
        raise ValueError('Scan timestamps differ by more than one microsecond')
    if not np.allclose(est[0, 1:], ref[0, 1:], atol=1e-8, rtol=0):
        raise ValueError('Trajectories must share the initial pose in the map frame')
    est_xy, ref_xy = est[:, 1:3], ref[:, 1:3]
    est_yaw, ref_yaw = est[:-1, 3], ref[:-1, 3]
    est_step = rotate(np.diff(est_xy, axis=0), -est_yaw)
    ref_step = rotate(np.diff(ref_xy, axis=0), -ref_yaw)
    body_error = est_step - ref_step
    # Rotate the translation error by estimated heading so the two terms sum exactly.
    translation_term = rotate(body_error, est_yaw)
    heading_term = rotate(ref_step, est_yaw) - np.diff(ref_xy, axis=0)
    delta_error = np.diff(est_xy - ref_xy, axis=0)
    residual = np.max(np.abs(np.cumsum(translation_term + heading_term - delta_error, axis=0)))
    sections = []
    for a, b in ((0, 100), (100, 200), (200, 299), (0, 299)):
        sl = slice(a, b)
        sections.append(dict(frames=[150+a, 150+b], steps=b-a,
                             body_error_sum_m=body_error[sl].sum(axis=0).tolist(),
                             translation_term_map_m=translation_term[sl].sum(axis=0).tolist(),
                             heading_term_map_m=heading_term[sl].sum(axis=0).tolist(),
                             measured_error_change_m=delta_error[sl].sum(axis=0).tolist()))
    starts = np.arange(0, 290, 10)
    ends = starts + 10
    windows = (rotate(est_xy[ends]-est_xy[starts], -est_yaw[starts])
               - rotate(ref_xy[ends]-ref_xy[starts], -ref_yaw[starts]))
    i = np.linalg.norm(windows, axis=1).argmax()
    return dict(sections=sections, max_reconstruction_residual_m=float(residual),
                final_error_with_reference_heading_m=rotate(body_error, ref_yaw).sum(axis=0).tolist(),
                worst_ten_step_frames=[int(150+starts[i]), int(150+ends[i])],
                worst_ten_step_body_error_m=windows[i].tolist(), full_windows=len(starts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('estimate')
    parser.add_argument('reference')
    args = parser.parse_args()
    print(json.dumps(analyze(read_csv(args.estimate), read_csv(args.reference)), indent=2, allow_nan=False))
