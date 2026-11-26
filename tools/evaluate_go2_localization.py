"""Independent check of Go2 localization on the official inspection (#185).

Every trajectory is compared on the same inspection scans without a ground truth:

* status counts, the longest run without a map match, and the same numbers inside
  control windows fixed in docs/go2-revisit-check.md before the comparison;
* revisits: scan pairs at least MIN_REVISIT_S apart whose relative pose is measured
  directly by scan-to-scan registration.  The reference is accepted only when
  registrations from different starting guesses converge to the same pose, so it does
  not inherit the starting guess of any one trajectory;
* a false anchor is a revisit pair where the trajectory reports both scans as matched
  to the map but its relative pose disagrees with the reference by more than
  FALSE_ANCHOR_M or FALSE_ANCHOR_DEG.

Only planar x, y and yaw are compared, so the 2D method and the 3D tracker share one
measure.  The relative pose does not depend on the map frame or on the start pose.
"""

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from mgts_nav.localize3d import Pose, SurfaceMap, scans, voxel_centroids

MIN_REVISIT_S = 15.0
REVISIT_RADIUS_M = 2.5
FALSE_ANCHOR_M = 0.30
FALSE_ANCHOR_DEG = 5.0
AGREE_M = 0.05
AGREE_DEG = 1.0
REFERENCE_INLIER_M = 0.10
REFERENCE_MIN_INLIERS = 0.5
SUBMAP_HALF = 2
ANCHOR_STEP = 10
TURN_DEG_S = 20.0
WINDOWS = {
    'start_0_10s': (0.0, 10.0),
    'event_35_50s': (35.0, 50.0),
    'end_95s_on': (95.0, 1e9),
}
# Extra starting guesses around the odometry guess that do not come from any method.
PERTURBATIONS = [(0.3, 0.0, 0.0), (-0.3, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, -0.3, 0.0),
                 (0.0, 0.0, np.radians(10)), (0.0, 0.0, -np.radians(10))]
DISTINCT_GUESS_M = 0.20
DISTINCT_GUESS_DEG = 5.0


def planar(xyz, rotation):
    yaw = rotation.as_euler('ZYX')[0]
    return np.array([xyz[0], xyz[1], yaw])


def relative(a, b):
    """Pose of b in the frame of a, planar."""
    c, s = np.cos(a[2]), np.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])])


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def read_track(path):
    """Read t, x, y, yaw/quaternion and status from a trajectory CSV."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'{path}: empty trajectory')
    times = np.array([float(r['t']) for r in rows])
    if 'yaw' in rows[0]:
        yaw = np.array([float(r['yaw']) for r in rows])
    else:
        quat = np.array([[float(r[k]) for k in ('qx', 'qy', 'qz', 'qw')] for r in rows])
        yaw = Rotation.from_quat(quat).as_euler('ZYX')[:, 0]
    poses = np.column_stack([[float(r['x']) for r in rows], [float(r['y']) for r in rows], yaw])
    status = np.array([r.get('status', 'matched') for r in rows])
    return times, poses, status


def on_scans(scan_times, track, tolerance=1e-3):
    """Rows of a trajectory at the scan stamps; missing scans become nan/'missing'."""
    times, poses, status = track
    index = np.clip(np.searchsorted(times, scan_times), 0, len(times) - 1)
    left = np.clip(index - 1, 0, len(times) - 1)
    index = np.where(np.abs(times[left] - scan_times) < np.abs(times[index] - scan_times), left, index)
    found = np.abs(times[index] - scan_times) <= tolerance
    out = np.full((len(scan_times), 3), np.nan)
    out[found] = poses[index[found]]
    labels = np.where(found, status[index], 'missing')
    return out, labels


def longest_unmatched(times, labels, period):
    best = (0, 0.0, None)
    start = None
    for i, label in enumerate(list(labels) + ['matched']):
        if label != 'matched' and start is None:
            start = i
        elif label == 'matched' and start is not None:
            seconds = times[i - 1] - times[start] + period
            if seconds > best[1]:
                best = (i - start, seconds, start)
            start = None
    count, seconds, first = best
    return {'scans': count, 'seconds': round(seconds, 3),
            'starts_s': None if first is None else round(times[first] - times[0], 3)}


def status_summary(times, labels, period, mask=None):
    mask = np.ones(len(times), bool) if mask is None else mask
    t, lab = times[mask], labels[mask]
    if not len(t):
        return {'scans': 0}
    return {'scans': int(len(t)),
            'matched': int(np.count_nonzero(lab == 'matched')),
            'matched_fraction': round(float(np.mean(lab == 'matched')), 4),
            'longest_unmatched': longest_unmatched(t, lab, period)}


def turn_mask(times, odom_poses, rate_deg_s=TURN_DEG_S, half_window_s=0.1):
    mask = np.zeros(len(times), bool)
    for i, t in enumerate(times):
        lo = np.searchsorted(times, t - half_window_s)
        hi = np.searchsorted(times, t + half_window_s, side='right') - 1
        if hi > lo:
            turn = abs(wrap(odom_poses[hi, 2] - odom_poses[lo, 2]))
            mask[i] = np.degrees(turn) / (times[hi] - times[lo]) >= rate_deg_s
    return mask


def submap(clouds, odom, index):
    """Neighbouring deskewed scans merged by odometry, in the body frame of `index`."""
    ref = odom[index]
    lo, hi = max(0, index - SUBMAP_HALF), min(len(clouds), index + SUBMAP_HALF + 1)
    parts = [ref.rotation.inv().apply(odom[k].apply(clouds[k]) - ref.xyz) for k in range(lo, hi)]
    return voxel_centroids(np.concatenate(parts), 0.05)


def lift(rel, odom_a, odom_b):
    """3D guess for b in the frame of a: planar part from `rel`, tilt and height from odometry."""
    odo = Pose(odom_a.rotation.inv() * odom_b.rotation,
               odom_a.rotation.inv().apply(odom_b.xyz - odom_a.xyz))
    roll_pitch = Rotation.from_euler('ZYX', [0, *odo.rotation.as_euler('ZYX')[1:]])
    return Pose(Rotation.from_euler('Z', rel[2]) * roll_pitch, np.array([rel[0], rel[1], odo.xyz[2]]))


def close(a, b, metres, degrees):
    return np.hypot(*(a[:2] - b[:2])) <= metres and np.degrees(abs(wrap(a[2] - b[2]))) <= degrees


def reference_pose(target, source, guesses):
    """Register source onto target from several guesses; accept only an agreed, tight fit.

    Agreement needs two converged fits whose starting guesses were clearly different,
    so two methods that happen to report the same pose do not confirm each other.
    """
    surface = SurfaceMap(target)
    fits = []
    for name, guess in guesses.items():
        pose, ok = surface.align(source, guess)
        if not ok:
            continue
        distances, _ = surface.tree.query(pose.apply(source))
        inliers = float(np.mean(distances < REFERENCE_INLIER_M))
        if inliers >= REFERENCE_MIN_INLIERS:
            fits.append((name, planar(pose.xyz, pose.rotation), inliers, planar(guess.xyz, guess.rotation)))
    for _, a, _, _ in fits:
        group = [f for f in fits if close(a, f[1], AGREE_M, AGREE_DEG)]
        distinct = any(not close(f[3], g[3], DISTINCT_GUESS_M, DISTINCT_GUESS_DEG)
                       for f in group for g in group)
        if distinct:
            best = max(group, key=lambda f: f[2])
            return best[1], [f[0] for f in group], best[2]
    return None, [f[0] for f in fits], None


def revisit_pairs(times, tracks):
    """Pairs (i, j) at least MIN_REVISIT_S apart that are close under any trajectory."""
    pairs = set()
    for poses in tracks.values():
        for i in range(0, len(times), ANCHOR_STEP):
            if not np.isfinite(poses[i, 0]):
                continue
            later = np.flatnonzero(times >= times[i] + MIN_REVISIT_S)
            later = later[np.isfinite(poses[later, 0])]
            if not len(later):
                continue
            distance = np.hypot(*(poses[later, :2] - poses[i, :2]).T)
            k = int(np.argmin(distance))
            if distance[k] <= REVISIT_RADIUS_M:
                pairs.add((i, int(later[k])))
    return sorted(pairs)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def evaluate(inspection, track_paths):
    stream = list(scans(inspection))
    times = np.array([s[0] for s in stream])
    clouds = [s[1] for s in stream]
    odom = [s[2] for s in stream]
    period = float(np.median(np.diff(times)))
    tracks = {'odometry': (times, np.array([planar(p.xyz, p.rotation) for p in odom]),
                           np.array(['odometry'] * len(times)))}
    for name, path in track_paths.items():
        tracks[name] = read_track(path)
    on = {name: on_scans(times, track) for name, track in tracks.items()}
    poses = {name: value[0] for name, value in on.items()}
    labels = {name: value[1] for name, value in on.items()}

    masks = {name: (times - times[0] >= lo) & (times - times[0] < hi) for name, (lo, hi) in WINDOWS.items()}
    masks['turns_ge_20deg_s'] = turn_mask(times, poses['odometry'])
    report = {'scans': len(times), 'period_s': round(period, 5), 't0': times[0], 'windows': {},
              'status': {name: status_summary(times, labels[name], period) for name in tracks}}
    for window, mask in masks.items():
        report['windows'][window] = {name: status_summary(times, labels[name], period, mask) for name in tracks}

    pairs = revisit_pairs(times, poses)
    rows = []
    for i, j in pairs:
        guesses = {}
        for name in tracks:
            if np.all(np.isfinite(poses[name][[i, j], 0])):
                guesses[name] = lift(relative(poses[name][i], poses[name][j]), odom[i], odom[j])
        base = relative(poses['odometry'][i], poses['odometry'][j])
        for k, offset in enumerate(PERTURBATIONS):
            guesses[f'perturbed{k}'] = lift(base + np.array(offset), odom[i], odom[j])
        # Target scan i, source scan j: the fitted pose is j in the frame of i.
        reference, agreed, inliers = reference_pose(submap(clouds, odom, i), submap(clouds, odom, j), guesses)
        row = {'i': i, 'j': j, 'ti_s': round(times[i] - times[0], 3), 'tj_s': round(times[j] - times[0], 3),
               'reference': None if reference is None else [round(float(v), 4) for v in reference],
               'agreed': agreed, 'inliers': None if inliers is None else round(inliers, 3)}
        for name in tracks:
            if reference is None or name not in guesses:
                continue
            delta = relative(poses[name][i], poses[name][j]) - reference
            row[name] = {'error_m': round(float(np.hypot(*delta[:2])), 4),
                         'error_deg': round(float(np.degrees(abs(wrap(delta[2])))), 3),
                         'anchored': bool(labels[name][i] == 'matched' and labels[name][j] == 'matched')}
        rows.append(row)

    checked = [r for r in rows if r['reference'] is not None]
    report['revisits'] = {'candidates': len(rows), 'with_reference': len(checked), 'methods': {}}
    for name in tracks:
        errors = [r[name] for r in checked if name in r]
        if not errors:
            continue
        metres = np.array([e['error_m'] for e in errors])
        degrees = np.array([e['error_deg'] for e in errors])
        bad = (metres > FALSE_ANCHOR_M) | (degrees > FALSE_ANCHOR_DEG)
        anchored = np.array([e['anchored'] for e in errors])
        report['revisits']['methods'][name] = {
            'pairs': len(errors),
            'median_m': round(float(np.median(metres)), 4), 'p95_m': round(float(np.percentile(metres, 95)), 4),
            'max_m': round(float(metres.max()), 4),
            'median_deg': round(float(np.median(degrees)), 3), 'p95_deg': round(float(np.percentile(degrees, 95)), 3),
            'over_threshold': int(bad.sum()),
            'false_anchors': int((bad & anchored).sum()),
            'anchored_pairs': int(anchored.sum())}
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inspection', type=Path)
    parser.add_argument('--track', action='append', default=[], metavar='NAME=CSV',
                        help='trajectory to evaluate; repeat for several methods')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    tracks = dict(item.split('=', 1) for item in args.track)
    args.out.mkdir(parents=True)
    report, rows = evaluate(args.inspection, tracks)
    report['inputs'] = {
        'inspection': {p.name: sha256(p) for p in sorted(args.inspection.glob('*.mcap'))},
        'tracks': {name: {'path': str(path), 'sha256': sha256(path)} for name, path in tracks.items()},
        'commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
        'thresholds': {'false_anchor_m': FALSE_ANCHOR_M, 'false_anchor_deg': FALSE_ANCHOR_DEG,
                       'min_revisit_s': MIN_REVISIT_S, 'revisit_radius_m': REVISIT_RADIUS_M,
                       'agree_m': AGREE_M, 'agree_deg': AGREE_DEG, 'windows_s': WINDOWS}}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (args.out / 'revisits.json').write_text(json.dumps(rows, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'revisits')}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
