"""Evaluate MGTS visual yaw on fixed TUM Freiburg 1 rpy pairs with motion-capture poses."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, map_coordinates
from scipy.spatial.transform import Rotation, Slerp

from mgts_nav.image_pairs import features, tracks
from mgts_nav.vision import estimate


K = np.array([[517.3, 0., 318.6], [0., 516.5, 255.3], [0., 0., 1.]])
MOUNT = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])


def rectification():
    v, u = np.indices((480, 640), dtype=float)
    x, y = (u-K[0, 2])/K[0, 0], (v-K[1, 2])/K[1, 1]
    k1, k2, p1, p2, k3 = .2624, -.9531, -.0054, .0026, 1.1633
    r2 = x*x + y*y
    scale = 1 + r2*(k1 + r2*(k2 + r2*k3))
    sx = (x*scale + 2*p1*x*y + p2*(r2+2*x*x))*K[0, 0]+K[0, 2]
    sy = (y*scale + p1*(r2+2*y*y) + 2*p2*x*y)*K[1, 1]+K[1, 2]
    valid = (sx >= 0) & (sx < 639) & (sy >= 0) & (sy < 479)
    return np.stack((sy, sx)), binary_erosion(valid, structure=np.ones((11, 11), bool))


def frame(path, coords, valid):
    with Image.open(path) as image:
        if image.size != (640, 480) or image.mode != 'RGB':
            raise ValueError(f'Unexpected Freiburg 1 RGB frame: {path}')
        gray = np.asarray(image) @ [.299, .587, .114]
    pts, descriptors = features(map_coordinates(gray, coords, order=1, mode='constant'))
    inside = valid[pts[:, 1], pts[:, 0]]
    return pts[inside], descriptors[inside]


def depth_tracks(rows, depth, coords):
    kept, nearest = [], np.inf
    for row in rows:
        sy, sx = np.rint(coords[:, row[2], row[1]]).astype(int)
        patch = depth[sy-1:sy+2, sx-1:sx+2]
        if patch.size == 9 and patch.min() > 0:
            kept.append(row)
            nearest = min(nearest, patch.min() / 5000.)
    return kept, float(nearest-.10) if kept and nearest > .10 else None


def analyze(data, output):
    output.mkdir(parents=True, exist_ok=False)
    rgb, depths = [np.array([line.split() for line in (data/name).read_text(encoding='ascii').splitlines()
                             if line.strip() and not line.startswith('#')]) for name in ('rgb.txt', 'depth.txt')]
    rt, dt = rgb[:, 0].astype(float), depths[:, 0].astype(float)
    ref = np.loadtxt(data/'groundtruth.txt')
    times = ref[:, 0]
    slerp = Slerp(times, Rotation.from_quat(ref[:, 4:8]))
    coords, valid = rectification()
    records = []
    with (output/'pairs.jsonl').open('x', encoding='utf-8') as pairs:
        for i in range(0, len(rgb)-6, 10):
            first = frame(data/rgb[i, 1], coords, valid)
            j = int(np.argmin(abs(dt-rt[i])))
            with Image.open(data/depths[j, 1]) as image:
                depth = np.array(image)
            for gap, max_dt in [(1, .10), (6, .40)]:
                t = rt[[i, i+gap]]
                rec = dict(index=i, gap=gap, t0=t[0], t1=t[1])
                records.append(rec)
                if not 0 < t[1]-t[0] <= max_dt:
                    rec['reason'] = 'rgb_interval'
                    continue
                hi = np.searchsorted(times, t)
                if (hi.min() == 0 or hi.max() == len(times) or np.any(times[hi]-times[hi-1] > .04)):
                    rec['reason'] = 'mocap_gap'
                    continue
                if abs(dt[j]-t[0]) > .02:
                    rec['reason'] = 'depth_gap'
                    continue
                position = np.array([np.interp(t, times, ref[:, col]) for col in (1, 2, 3)])
                motion = float(np.linalg.norm(position[:, 1]-position[:, 0])+.002)
                rot0, rot1 = slerp(t).as_matrix()
                delta = MOUNT @ rot0.T @ rot1 @ MOUNT.T
                yaw = float(np.arctan2(delta[1, 0], delta[0, 0]))
                tilt = float(np.rad2deg(np.arccos(np.clip(delta[2, 2], -1, 1))))
                second = frame(data/rgb[i+gap, 1], coords, valid)
                matched = tracks(first, second)
                kept, distance = depth_tracks(matched, depth, coords)
                pair = dict(t0=t[0], t1=t[1], K=K.tolist(), base_from_camera_xyzw=Rotation.from_matrix(MOUNT).as_quat().tolist(),
                            camera_motion_bound_m=motion, min_scene_range_m=distance, tracks=kept)
                pairs.write(json.dumps(pair, allow_nan=False)+'\n')
                result = estimate(pair)
                parallax = None if distance is None else bool(motion/distance > np.sin(np.deg2rad(.35)))
                rec.update(reason=result['reason'], tracks=len(matched), depth_tracks=len(kept),
                           reference_deg=float(np.rad2deg(yaw)), tilt_deg=tilt, parallax_violation=parallax,
                           motion_bound_m=motion, range_bound_m=distance)
                if result['accepted']:
                    error = result['yaw_rad']-yaw
                    rec.update(estimate_deg=float(np.rad2deg(result['yaw_rad'])),
                               error_deg=float(np.rad2deg(np.arctan2(np.sin(error), np.cos(error)))))
    summary = {}
    for gap in (1, 6):
        group = [r for r in records if r['gap'] == gap]
        accepted = [r for r in group if 'error_deg' in r]
        errors = np.array([r['error_deg'] for r in accepted])
        summary[gap] = dict(total=len(group), accepted=len(accepted), reasons=Counter(r['reason'] for r in group),
                            tilt_over_3=sum(r.get('tilt_deg', 0) > 3 for r in group),
                            parallax_over_gate=sum(r.get('parallax_violation') is True for r in group),
                            accepted_outside=sum(r['tilt_deg'] > 3 or r['parallax_violation'] for r in accepted),
                            rms_deg=float(np.sqrt(np.mean(errors**2))) if errors.size else None,
                            max_abs_deg=float(np.max(abs(errors))) if errors.size else None)
    (output/'results.json').write_text(json.dumps(dict(summary=summary, pairs=records), indent=2, allow_nan=False),
                                       encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sequence', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    analyze(args.sequence, args.output)
