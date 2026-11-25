"""Self-check of a sensor profile on the mapping recording, without any reference.

    python -m mgts_nav.self_check BAG --config PROFILE.json [--out REPORT.json] [--budget 60]

Errors that leave scan matching at 100 % are looked for by consistency alone. The
recording is read once; every hypothesis runs the unchanged bag.scans() and slam.track().

* mirror (lidar upside down, roll 180 deg missing): the flipped profile is built too and
  wins on map consistency, or the lidar trajectory turns against odometry/IMU.
* lidar pitch/roll: a small grid of extra rotations. A wrong tilt turns the floor or the
  ceiling into points inside the height band. Map consistency (see consistency()) drops,
  because these points are not seen again from other places or the lidar saw through them.
* lidar yaw: the travel direction of the lidar trajectory and of odometry in the body frame.
* clock offset of odometry or IMU: the lag that maximises the yaw-rate correlation between
  the lidar trajectory and the sensor (as tools/clock_lag.py, without a separate map run).

A finding is reported only when it clearly beats the claimed profile; otherwise the
profile is confirmed. The report says which checks could not run (no IMU, no odometry,
no turns). Exit code: 0 no inconsistency, 1 warning, 2 input error.
"""
import argparse
import copy
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import bag as bagmod
from .slam import rotation, track

TIME_SEPARATION_S = 4.0   # "another visit": scans at least this far apart in time
SUPPORT_RADIUS_M = 0.10
CELL_M = 0.05
FREE_CELL_M = 0.10
FREE_STEP_M = 0.05
FREE_MARGIN_M = 0.30


class MemoryBag:
    """In-memory stand-in for rosbags AnyReader with already deserialized messages."""

    def __init__(self, connections, messages):
        self.connections = connections
        self._messages = messages

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def messages(self, connections=()):
        wanted = {id(c) for c in connections}
        for conn, t, msg in self._messages:
            if id(conn) in wanted:
                yield conn, t, msg

    def deserialize(self, msg, msgtype):
        return msg


def load_recording(path, cfg, max_scans=400):
    """Read the profile topics once. Long recordings keep every k-th lidar message."""
    with bagmod.reader(path) as bag:
        conns = bagmod.connections(bag, cfg)
        selected = [c for group in conns.values() for c in group]
        lidar_count = sum(c.msgcount for c in conns['lidar'])
        stride = max(1, int(np.ceil(lidar_count / max_scans)))
        messages, seen = [], 0
        lidar_ids = {id(c) for c in conns['lidar']}
        for conn, t, raw in bag.messages(connections=selected):
            if id(conn) in lidar_ids:
                seen += 1
                if (seen - 1) % stride:
                    continue
            messages.append((conn, t, bag.deserialize(raw, conn.msgtype)))
        return MemoryBag(list(bag.connections), messages), stride


@contextmanager
def _memory_reader(memory):
    original = bagmod.reader
    bagmod.reader = lambda path: memory
    try:
        yield
    finally:
        bagmod.reader = original


def with_rotation(cfg, delta_rpy_deg):
    """Profile with an extra base-frame rotation applied after the claimed lidar rotation."""
    out = copy.deepcopy(cfg)
    delta = Rotation.from_euler('xyz', delta_rpy_deg, degrees=True)
    out['base_from_lidar_xyzw'] = (delta * Rotation.from_quat(cfg['base_from_lidar_xyzw'])).as_quat().tolist()
    return out


class _NoGrid:
    """track() builds its own keyframe map; the occupancy grid is not needed here."""

    def observe(self, origin, endpoints):
        pass


def build(memory, cfg):
    """Run the unchanged scan pipeline and SLAM; return scans and estimates."""
    with _memory_reader(memory):
        scans = list(bagmod.scans('<memory>', cfg))
    estimates = list(track(iter(scans), np.zeros(3), grid=_NoGrid()))
    return scans, estimates


def _key(cells):
    return cells[:, 0] * 1_000_003 + cells[:, 1]


def _time_index(key, times):
    """Sorted cell keys with the first and last time each cell was seen."""
    order = np.argsort(key, kind='stable')
    unique, start = np.unique(key[order], return_index=True)
    return unique, np.minimum.reduceat(times[order], start), np.maximum.reduceat(times[order], start)


def _seen_apart(index, cells, times, reach, separation):
    """True where a cell within `reach` cells was seen at least `separation` s away in time."""
    unique, first, last = index
    hit = np.zeros(len(times), bool)
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            if dx * dx + dy * dy > reach * reach:
                continue
            probe = _key(cells + [dx, dy])
            idx = np.clip(np.searchsorted(unique, probe), 0, len(unique) - 1)
            ok = unique[idx] == probe
            hit[ok] |= ((times[ok] - first[idx[ok]] >= separation)
                        | (last[idx[ok]] - times[ok] >= separation))
    return hit


def _count(keys, probe):
    """How many entries of `keys` equal each probe key."""
    unique, counts = np.unique(keys, return_counts=True)
    idx = np.clip(np.searchsorted(unique, probe), 0, len(unique) - 1)
    return np.where(unique[idx] == probe, counts[idx], 0)


def consistency(scans, estimates, separation=TIME_SEPARATION_S):
    """How well the map points agree with the other scans of the same recording.

    confirmed: a hit within SUPPORT_RADIUS_M from a scan taken at least `separation` s
    apart. crossed: more scans saw through the point's cell (free part of a ray, stopping
    FREE_MARGIN_M short of its end) than hit it. A phantom wall made of floor returns moves
    with the robot, so it is not confirmed, or it fills an area that other rays cross.
    Comparing the counts, instead of "crossed at all", keeps real walls seen at grazing
    angles in a corridor. score = share of points that are confirmed and not crossed.
    """
    rows = []
    for scan, est in zip(scans, estimates):
        if est.status == 'prediction_only' or not len(scan.xy):
            continue
        c, s = np.cos(est.pose[2]), np.sin(est.pose[2])
        turn = np.array([[c, -s], [s, c]])
        rows.append((scan.t, scan.xy @ turn.T + est.pose[:2], turn @ scan.origin + est.pose[:2]))
    if len(rows) < 3:
        return dict(points=0, score=0.0, confirmed=0.0, crossed=0.0, scans_used=len(rows))
    times = np.concatenate([np.full(len(xy), t) for t, xy, _ in rows])
    points = np.vstack([xy for _, xy, _ in rows])
    cells = np.floor(points / CELL_M).astype(np.int64)
    confirmed = _seen_apart(_time_index(_key(cells), times), cells, times,
                            int(round(SUPPORT_RADIUS_M / CELL_M)), separation)
    # One key per cell and scan: counts below are numbers of scans.
    free_keys, hit_keys = [], []
    for _, xy, origin in rows:
        ray = xy - origin
        length = np.linalg.norm(ray, axis=1)
        hit_keys.append(np.unique(_key(np.floor(xy / FREE_CELL_M).astype(np.int64))))
        steps = np.arange(FREE_STEP_M, length.max(), FREE_STEP_M)
        if not len(steps):
            continue
        keep = steps[None, :] < (length - FREE_MARGIN_M)[:, None]
        unit = ray / np.maximum(length, 1e-9)[:, None]
        samples = origin + unit[:, None, :] * steps[None, :, None]
        free_keys.append(np.unique(_key(np.floor(samples[keep] / FREE_CELL_M).astype(np.int64))))
    probe = _key(np.floor(points / FREE_CELL_M).astype(np.int64))
    crossed = np.zeros(len(times), bool)
    if free_keys:
        crossed = _count(np.concatenate(free_keys), probe) > _count(np.concatenate(hit_keys), probe)
    good = confirmed & ~crossed
    return dict(points=int(len(times)), score=float(good.mean()), confirmed=float(confirmed.mean()),
                crossed=float(crossed.mean()), scans_used=len(rows))


def sensor_series(memory, cfg, role):
    """Header times (with the profile offset) and yaw rate of odometry or IMU in the base frame.

    For odometry also returns positions and headings for the body-frame motion check.
    """
    offset = cfg.get('header_offset_s', {}).get(role, 0.0)
    conns = [c for c in memory.connections if c.topic == cfg[role]]
    rows = []
    for _, _, m in memory.messages(connections=conns):
        if role == 'odom':
            p, q = m.pose.pose.position, m.pose.pose.orientation
            heading = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            rows.append([bagmod.stamp(m) + offset, p.x, p.y, np.arctan2(heading[1, 0], heading[0, 0])])
        else:
            w = m.angular_velocity
            rows.append([bagmod.stamp(m) + offset,
                         Rotation.from_quat(cfg['base_from_imu_xyzw']).apply([w.x, w.y, w.z])[2]])
    a = np.asarray(rows, float)
    if len(a) < 10:
        return None
    a = a[np.r_[True, np.diff(a[:, 0]) > 0]]
    if role == 'imu':
        return dict(t=a[:, 0], rate=a[:, 1])
    yaw = np.unwrap(a[:, 3])
    return dict(t=a[1:, 0] - np.diff(a[:, 0]) / 2, rate=np.diff(yaw) / np.diff(a[:, 0]),
                pose_t=a[:, 0], pose=np.column_stack([a[:, 1:3], yaw]))


def lidar_rate(estimates):
    good = [(e.t, e.pose[2]) for e in estimates if e.status != 'prediction_only']
    a = np.asarray(good, float)
    if len(a) < 10:
        return None
    return a[1:, 0] - np.diff(a[:, 0]) / 2, np.diff(a[:, 1]) / np.diff(a[:, 0])


def correlate(ref_t, ref_rate, t, rate, max_lag=1.0, step=0.005, max_hole=0.15):
    """Correlation of yaw rates against the lag of the sensor (positive: sensor stamps late)."""
    lags = np.arange(-max_lag, max_lag + step / 2, step)
    grid = np.arange(ref_t[0], ref_t[-1], 0.01)
    reference = np.interp(grid, ref_t, ref_rate)
    scores = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        probe = grid + lag
        index = np.clip(np.searchsorted(t, probe), 1, len(t) - 1)
        hole = np.minimum(np.abs(t[index] - probe), np.abs(t[index - 1] - probe)) > max_hole
        ok = ~hole & (probe >= t[0]) & (probe <= t[-1])
        if ok.sum() < 0.5 * len(grid):
            continue
        a, b = reference[ok], np.interp(probe[ok], t, rate)
        a, b = a - a.mean(), b - b.mean()
        den = np.sqrt((a @ a) * (b @ b))
        if den > 0:
            scores[i] = a @ b / den
    return lags, scores


def heading_offset(estimates, odom, lidar_xy, window_s=1.0, min_travel_m=0.2):
    """Yaw error of the lidar calibration from travel directions in the body frame.

    With an extra lidar yaw e in the profile, the matched "base" heading is the true one
    minus e and its position is the true position of the point c = t - R(-e) t, where t is
    the lidar offset in the base. Over a window with odometry displacement d and turn dth,
    the lidar displacement in its own body frame is R(e) (d + (R(dth) - I) c). e is found
    by a few fixed-point steps; the lever arm matters for a lidar mounted backwards.
    """
    good = [e for e in estimates if e.status != 'prediction_only']
    t = np.array([e.t for e in good])
    poses = np.array([e.pose for e in good])
    pairs = []
    for i in range(len(t)):
        j = np.searchsorted(t, t[i] + window_s)
        if j >= len(t) or t[j] - t[i] > 2 * window_s:
            continue
        if not (odom['pose_t'][0] <= t[i] and t[j] <= odom['pose_t'][-1]):
            continue
        a = np.array([np.interp(t[i], odom['pose_t'], col) for col in odom['pose'].T])
        b = np.array([np.interp(t[j], odom['pose_t'], col) for col in odom['pose'].T])
        d_odom = rotation(a[2]).T @ (b[:2] - a[:2])
        d_lidar = rotation(poses[i, 2]).T @ (poses[j, :2] - poses[i, :2])
        if np.linalg.norm(d_odom) < min_travel_m or np.linalg.norm(d_lidar) < min_travel_m:
            continue
        pairs.append((d_odom, d_lidar, b[2] - a[2]))
    if len(pairs) < 5:
        return None
    lever = np.asarray(lidar_xy, float)
    error = 0.0
    for _ in range(6):
        c = lever - rotation(-error) @ lever
        angles = []
        for d_odom, d_lidar, turn in pairs:
            expected = d_odom + (rotation(turn) - np.eye(2)) @ c
            angles.append(np.arctan2(expected[0] * d_lidar[1] - expected[1] * d_lidar[0], expected @ d_lidar))
        angles = np.asarray(angles)
        # Circular mean: a lidar mounted backwards gives angles near +-180 deg.
        error = float(np.angle(np.mean(np.exp(1j * angles))))
    spread = np.abs(np.angle(np.exp(1j * (angles - error))))
    return dict(samples=len(angles), offset_deg=float(np.degrees(error)),
                spread_deg=float(np.degrees(np.percentile(spread, 75))))


# Decision thresholds, set on correct recordings (docs/self-check.md).
TILT_MARGIN = 0.04         # consistency gain over the claimed rotation that counts as an error
TILT_TIE = 0.005           # values within this of the best form the estimate
COARSE_DEG = (-20, -15, -10, -5, 5, 10, 15, 20)
REFINE_STEPS_DEG = (4, 2)
YAW_LIMIT_DEG = 3.5
YAW_MAX_SPREAD_DEG = 20.0
LAG_LIMIT_S = 0.05
MIN_CORRELATION = 0.7
MIRROR_CORRELATION = -0.5


class Budget:
    def __init__(self, seconds):
        self.start, self.seconds = time.monotonic(), seconds

    def left(self):
        return self.seconds - (time.monotonic() - self.start)


def _rotation_rpy(cfg):
    rpy = Rotation.from_quat(cfg['base_from_lidar_xyzw']).as_euler('xyz', degrees=True)
    return [round(float(v), 2) for v in rpy]


def _matched(estimates):
    return sum(e.status != 'prediction_only' for e in estimates)


def sweep_tilt(memory, cfg, base, budget, log):
    """Coordinate search over an extra roll and pitch of the lidar in the base frame.

    Each candidate rebuilds the trajectory from the same scans and is scored by
    consistency(). Order: a pitch line, a roll line, then a compass search over both.
    Tilting further towards the floor only drops far points, so the score has a plateau;
    among values within TILT_TIE of the best, the one keeping most consistent points wins.
    """
    base_score = base['score']
    cache = {(0.0, 0.0): base_score}
    kept = {(0.0, 0.0): base['score'] * base['points']}
    complete = True

    def score(roll, pitch):
        nonlocal complete
        key = (float(roll), float(pitch))
        if key not in cache:
            if budget.left() <= 0:
                complete = False
                return
            scans, estimates = build(memory, with_rotation(cfg, [roll, pitch, 0]))
            found = consistency(scans, estimates) if _matched(estimates) >= 3 else dict(score=0.0, points=0)
            cache[key], kept[key] = found['score'], found['score'] * found['points']
            log(f'  tilt roll {roll:+.0f} pitch {pitch:+.0f} deg: consistency {cache[key]:.3f}')

    def line(best, axis):
        return {k[axis]: s for k, s in cache.items() if k[1 - axis] == best[1 - axis]}

    # Lines through the claimed rotation: pitch, then roll at the best pitch.
    best = [0.0, 0.0]
    for axis in (1, 0):
        for v in COARSE_DEG:
            probe = list(best)
            probe[axis] = v
            score(*probe)
        values = line(best, axis)
        best[axis] = max(values, key=values.get)
    # Roll and pitch errors partly compensate each other, so finish with a compass
    # search over both (steps 4 and 2 deg) once the map clearly improved.
    if cache[tuple(best)] - base_score >= TILT_MARGIN:
        for step in REFINE_STEPS_DEG:
            moved = True
            while moved and complete:
                moved = False
                for dr, dp in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    probe = (best[0] + dr * step, best[1] + dp * step)
                    score(*probe)
                    if cache.get(probe, -1) > cache[tuple(best)] + 1e-3:
                        best, moved = list(probe), True
                        break
    gain = cache[tuple(best)] - base_score
    flagged = gain >= TILT_MARGIN
    estimate = [0.0, 0.0]
    if flagged:
        near = [k for k, s in cache.items() if s >= cache[tuple(best)] - TILT_TIE]
        estimate = list(max(near, key=kept.get))
    return dict(status='warning' if flagged else 'ok', claimed_consistency=round(base_score, 4),
                best_consistency=round(cache[tuple(best)], 4), gain=round(gain, 4),
                correction_roll_pitch_deg=estimate, evaluated=len(cache), complete=complete), estimate


def rate_check(memory, cfg, estimates, role):
    """Lag and sign of the yaw-rate correlation between the lidar trajectory and a sensor."""
    series = sensor_series(memory, cfg, role)
    lidar = lidar_rate(estimates)
    if series is None or lidar is None:
        return dict(status='not_measured', reason=f'too few {role} messages or matched scans')
    lags, scores = correlate(*lidar, series['t'], series['rate'])
    if not np.isfinite(scores).any():
        return dict(status='not_measured', reason=f'{role} does not overlap the lidar in time')
    i = int(np.nanargmax(np.abs(scores)))
    out = dict(lag_s=round(float(lags[i]), 3), peak_correlation=round(float(scores[i]), 3))
    if scores[i] <= MIRROR_CORRELATION:
        out.update(status='mirror', reason=f'lidar yaw turns against {role}')
    elif scores[i] < MIN_CORRELATION:
        out.update(status='not_measured', reason='weak correlation: too few turns')
    elif abs(lags[i]) >= LAG_LIMIT_S:
        current = cfg.get('header_offset_s', {}).get(role, 0.0)
        out.update(status='warning', header_offset_s=round(current - float(lags[i]), 3))
    else:
        out.update(status='ok')
    return out


def heading_check(memory, cfg, estimates):
    if cfg.get('odom') is None:
        return dict(status='not_measured', reason='no odometry')
    series = sensor_series(memory, cfg, 'odom')
    found = None if series is None else heading_offset(estimates, series, cfg['base_from_lidar_xyz_m'][:2])
    if found is None:
        return dict(status='not_measured', reason='too little travel')
    if found['spread_deg'] > YAW_MAX_SPREAD_DEG:
        return dict(status='not_measured', reason='travel directions disagree', **found)
    flagged = abs(found['offset_deg']) >= YAW_LIMIT_DEG
    # The lidar sees the base turned by -offset, so the extra lidar yaw is -offset.
    return dict(status='warning' if flagged else 'ok',
                correction_yaw_deg=round(-found['offset_deg'], 1) if flagged else 0.0, **found)


def self_check(path, cfg, budget_s=60.0, max_scans=150, log=print):
    """Run all checks; return the report and the profile with the suggested corrections."""
    budget = Budget(budget_s)
    memory, stride = load_recording(path, cfg, max_scans)
    report = dict(bag=str(path), scan_stride=stride, claimed_lidar_rpy_deg=_rotation_rpy(cfg), checks={})
    checks = report['checks']
    scans, estimates = build(memory, cfg)
    base = consistency(scans, estimates)
    report['claimed'] = dict(scans=len(estimates), matched=_matched(estimates), **base)
    log(f'claimed profile: {_matched(estimates)}/{len(estimates)} scans matched, '
        f'map consistency {base["score"]:.3f}')
    fixed = copy.deepcopy(cfg)
    roles = [r for r in ('imu', 'odom') if cfg.get(r) is not None]
    # 1. Mirror: the lidar trajectory turns against odometry/IMU (lidar upside down).
    # The flipped hypothesis is built too: with a mirrored lidar the prediction from
    # odometry/IMU fights scan matching, and the map usually falls apart.
    rates = {r: rate_check(memory, cfg, estimates, r) for r in roles}
    flipped = with_rotation(cfg, [180, 0, 0])
    flip_scans, flip_estimates = build(memory, flipped)
    flip = consistency(flip_scans, flip_estimates)
    turns_against = any(m['status'] == 'mirror' for m in rates.values())
    evidence = dict(claimed_consistency=round(base['score'], 4), flipped_consistency=round(flip['score'], 4),
                    yaw_rate=rates)
    if (flip['score'] - base['score'] >= TILT_MARGIN
            or (turns_against and flip['score'] >= base['score'] - TILT_TIE)):
        fixed = flipped
        checks['mirror'] = dict(status='warning', correction_roll_deg=180, **evidence)
        log(f'MIRROR: lidar turned upside down fits better (consistency {base["score"]:.3f} -> '
            f'{flip["score"]:.3f}); extra roll 180 deg suggested')
        scans, estimates, base = flip_scans, flip_estimates, flip
    else:
        checks['mirror'] = dict(status='ok', **evidence)
    if _matched(estimates) < 10:
        report.update(verdict='not_measured', warnings=[], seconds=round(budget_s - budget.left(), 1))
        log('verdict: too few matched scans for a self-check; fix frames, height band and units first')
        return report, cfg
    # 2. Tilt: map consistency over an extra roll/pitch.
    tilt, delta = sweep_tilt(memory, fixed, base, budget, log)
    checks['tilt'] = tilt
    if tilt['status'] == 'warning':
        fixed = with_rotation(fixed, [delta[0], delta[1], 0])
        scans, estimates = build(memory, fixed)
        log(f'TILT: extra roll {delta[0]:+.1f} pitch {delta[1]:+.1f} deg raises map consistency '
            f'{tilt["claimed_consistency"]:.3f} -> {tilt["best_consistency"]:.3f}')
    # 3. Lidar yaw: travel direction of the lidar trajectory against odometry.
    checks['yaw'] = heading_check(memory, fixed, estimates)
    if checks['yaw']['status'] == 'warning':
        fixed = with_rotation(fixed, [0, 0, checks['yaw']['correction_yaw_deg']])
        log(f'YAW: lidar travel direction differs from odometry by {checks["yaw"]["offset_deg"]:+.1f} deg; '
            f'extra lidar yaw {checks["yaw"]["correction_yaw_deg"]:+.1f} deg suggested')
    # 4. Header clocks of odometry and IMU against the lidar.
    for role in roles:
        result = rate_check(memory, fixed, estimates, role)
        checks[f'clock_{role}'] = result
        if result['status'] == 'warning':
            fixed.setdefault('header_offset_s', {})[role] = result['header_offset_s']
            log(f'CLOCK: {role} headers lag the lidar by {result["lag_s"]:+.3f} s; '
                f'set header_offset_s.{role} = {result["header_offset_s"]}')
    warnings = [k for k, v in checks.items() if v['status'] == 'warning']
    report.update(verdict='warning' if warnings else 'ok', warnings=warnings,
                  suggested_lidar_rpy_deg=_rotation_rpy(fixed),
                  suggested={k: fixed[k] for k in ('base_from_lidar_xyzw', 'header_offset_s') if k in fixed},
                  complete=tilt['complete'], seconds=round(budget_s - budget.left(), 1))
    for name, value in checks.items():
        if value['status'] == 'not_measured':
            log(f'{name}: not measured ({value.get("reason", "")})')
    if not tilt['complete']:
        log('tilt search stopped by --budget; the tilt result covers only the evaluated values')
    log('verdict: ' + (f'WARNING {", ".join(warnings)}; check the suggested profile before mapping'
                       if warnings else 'no inconsistency found') + f' ({report["seconds"]} s)')
    return report, fixed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag', type=Path, help='Mapping recording')
    parser.add_argument('--config', required=True, type=Path, help='Sensor profile to check')
    parser.add_argument('--out', type=Path, help='Write the JSON report here')
    parser.add_argument('--suggest-profile', type=Path, help='Write the profile with the suggested corrections')
    parser.add_argument('--budget', type=float, default=60.0, help='Seconds for the tilt search (default 60)')
    parser.add_argument('--max-scans', type=int, default=150, help='Use at most this many lidar scans')
    parser.add_argument('-q', '--quiet', action='store_true', help='Do not print each tilt candidate')
    args = parser.parse_args()
    log = (lambda m: None if m.startswith('  ') else print(m)) if args.quiet else print
    try:
        cfg = bagmod.load_config(args.config)
        report, fixed = self_check(args.bag, cfg, args.budget, args.max_scans, log)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(2, f'{error}\n')
    report['profile'] = str(args.config)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if args.suggest_profile:
        if report['verdict'] != 'warning':
            print('no correction suggested; profile not written')
        else:
            original = json.loads(args.config.read_text(encoding='utf-8-sig'))
            original.update(report['suggested'])
            args.suggest_profile.write_text(json.dumps(original, indent=2), encoding='utf-8')
            print(f'suggested profile: {args.suggest_profile} (rebuild the map and compare before use)')
    sys.exit(1 if report['verdict'] == 'warning' else 0)


if __name__ == '__main__':
    main()
