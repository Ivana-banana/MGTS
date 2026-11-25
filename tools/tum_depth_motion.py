"""Compare two-frame RGB-D odometry to mocap on fixed Freiburg 1 rpy pairs."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from mgts_nav.image_pairs import tracks
from tools.tum_visual_yaw import K, MOUNT, frame, rectification


def unproject(uv, depth, coords):
    u, v = uv
    sy, sx = np.rint(coords[:, v, u]).astype(int)
    patch = depth[sy-1:sy+2, sx-1:sx+2]
    if patch.size != 9 or patch.min() == 0 or int(patch.max()) - int(patch.min()) > 500:
        return None
    z = float(np.median(patch)) / 5000
    return z * np.array([(u-K[0, 2])/K[0, 0], (v-K[1, 2])/K[1, 1], 1.])


def fit(source, target):
    a, b = source.mean(axis=0), target.mean(axis=0)
    u, s, vt = np.linalg.svd((source-a).T @ (target-b))
    if s[1] < 1e-10:
        return None
    sign = np.linalg.det(vt.T @ u.T)
    rot = vt.T @ np.diag([1., 1., np.sign(sign)]) @ u.T
    return rot, b-rot@a


def estimate(source, target):
    if len(source) < 8:
        return None
    rng = np.random.default_rng(17)
    best = None
    for _ in range(128):
        ids = rng.choice(len(source), 3, replace=False)
        hypothesis = fit(source[ids], target[ids])
        if hypothesis is None:
            continue
        rot, t = hypothesis
        err = np.linalg.norm(source @ rot.T + t - target, axis=1)
        mask = err <= .04
        if mask.sum() < max(8, .6*len(source)):
            continue
        score = (int(mask.sum()), -np.median(err[mask]))
        if best is None or score > best[0]:
            best = score, mask
    if best is None:
        return None
    model = fit(source[best[1]], target[best[1]])
    if model is not None:
        rot, t = model
        retained = int(np.sum(np.linalg.norm(source @ rot.T + t - target, axis=1) <= .04))
        if retained >= max(8, .6*len(source)):
            return rot, t, retained
    return None


def load_images(path):
    return [(float(t), name) for t, name in
            (line.split() for line in path.read_text().splitlines() if line.strip() and not line.startswith('#'))]


def run(data):
    rgb, depths = (load_images(data / f'{kind}.txt') for kind in ('rgb', 'depth'))
    depth_times = np.array([t for t, _ in depths])
    coords, valid = rectification()
    records = []
    for i in range(0, len(rgb)-6, 10):
        for gap, limit in ((1, .1), (6, .4)):
            t0, t1 = rgb[i][0], rgb[i+gap][0]
            row = dict(index=i, gap=gap, t0=t0, t1=t1)
            records.append(row)
            if not 0 < t1-t0 <= limit:
                row['reason'] = 'rgb_interval'
                continue
            ids = [int(np.argmin(abs(depth_times-t))) for t in (t0,t1)]
            if any(abs(depth_times[j]-t) > .02 for j,t in zip(ids, (t0,t1))):
                row['reason'] = 'depth_gap'
                continue
            clouds = [[], []]
            images = []
            for j in ids:
                with Image.open(data/depths[j][1]) as image:
                    images.append(np.asarray(image))
            pairs = tracks(frame(data/rgb[i][1], coords, valid), frame(data/rgb[i+gap][1], coords, valid))
            for _, u0, v0, u1, v1 in pairs:
                a = unproject((u0, v0), images[0], coords)
                b = unproject((u1, v1), images[1], coords)
                if a is not None and b is not None:
                    clouds[0].append(a)
                    clouds[1].append(b)
            row['depth_tracks'] = len(clouds[0])
            # Body motion T0^-1 T1 maps static points from the second frame into the first.
            result = estimate(np.asarray(clouds[1]), np.asarray(clouds[0]))
            if result is None:
                row['reason'] = 'too_few_tracks' if len(clouds[0]) < 8 else 'no_consensus'
                continue
            rot, translation, row['inliers'] = result
            row.update(reason='accepted', translation=(MOUNT @ translation).tolist(),
                       rotation=(MOUNT @ rot @ MOUNT.T).tolist())
    return records


def score(records, path):
    gt = np.loadtxt(path)
    rotations = Slerp(gt[:, 0], Rotation.from_quat(gt[:, 4:8]))
    summary = {}
    for gap in (1, 6):
        group = [r for r in records if r['gap'] == gap]
        errors, stationary_errors = [], []
        for row in group:
            if row['reason'] != 'accepted':
                continue
            t = [row['t0'], row['t1']]
            hi = np.searchsorted(gt[:, 0], t)
            if min(hi) == 0 or max(hi) == len(gt) or np.any(gt[hi, 0]-gt[hi-1, 0] > .04):
                continue
            r0, r1 = rotations(t).as_matrix()
            delta = MOUNT @ r0.T @ r1 @ MOUNT.T
            step = [np.diff(np.interp(t, gt[:, 0], gt[:, j]))[0] for j in (1, 2, 3)]
            translation = MOUNT @ r0.T @ step
            fitted = np.array(row['rotation'])
            reference_yaw = np.arctan2(delta[1, 0], delta[0, 0])
            yaw = np.arctan2(fitted[1, 0], fitted[0, 0]) - reference_yaw
            err = [np.linalg.norm(row['translation']-translation), np.rad2deg(np.arctan2(np.sin(yaw), np.cos(yaw))),
                   np.rad2deg(Rotation.from_matrix(fitted.T @ delta).magnitude())]
            row['errors_m_deg_deg'] = err
            errors.append(err)
            stationary_errors.append([np.linalg.norm(translation), np.rad2deg(reference_yaw),
                                      np.rad2deg(Rotation.from_matrix(delta).magnitude())])
        summary[gap] = dict(reasons=Counter(r['reason'] for r in group), scored=len(errors),
                            rms=np.sqrt(np.mean(np.square(errors), axis=0)).tolist() if errors else None,
                            stationary_rms=np.sqrt(np.mean(np.square(stationary_errors), axis=0)).tolist()
                            if errors else None)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sequence', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    records = run(args.sequence)
    summary = score(records, args.sequence/'groundtruth.txt')
    with args.output.open('x', encoding='utf-8') as dest:
        json.dump(dict(summary=summary, pairs=records), dest, indent=2, allow_nan=False)
    print(json.dumps(summary))
