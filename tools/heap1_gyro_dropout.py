"""Replay the HEAP-1 turning inspection with a timed loss of gyroscope measurements."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

from mgts_nav.bag import load_config, reader, scans, stamp
from mgts_nav.metrics import evaluate, read_csv


def mask_rates(source, target, topic, begin, end):
    store = get_typestore(Stores.ROS2_HUMBLE)
    count = 0
    with reader(source) as bag, Writer(target, version=8) as writer:
        conns = {c.id: writer.add_connection(c.topic, c.msgtype, typestore=store) for c in bag.connections}
        for conn, t, raw in bag.messages():
            if conn.topic == topic:
                msg = bag.deserialize(raw, conn.msgtype)
                if begin <= stamp(msg) < end:
                    msg.angular_velocity_covariance = msg.angular_velocity_covariance.copy()
                    msg.angular_velocity_covariance[0] = -1.
                    raw = store.serialize_cdr(msg, conn.msgtype)
                    count += 1
            writer.write(conns[conn.id], t, raw)
    if not count:
        raise ValueError('No IMU messages in selected window')
    return count


def fastest_second(source, cfg, start, stop):
    mount = Rotation.from_quat(cfg['base_from_imu_xyzw'])
    samples = []
    with reader(source) as bag:
        for conn, _, raw in bag.messages(connections=[c for c in bag.connections if c.topic == cfg['imu']]):
            msg = bag.deserialize(raw, conn.msgtype)
            if msg.angular_velocity_covariance[0] == -1 or msg.header.frame_id != cfg['imu_frame']:
                raise ValueError('Expected valid calibrated IMU stream')
            w = msg.angular_velocity
            samples.append((stamp(msg)-start, mount.apply([w.x, w.y, w.z])[2]))
    t, rate = np.asarray(samples).T
    span = stop-start
    if (t[0] > 0 or t[-1] < span or span < 1 or not np.isfinite(samples).all()
            or np.any(np.diff(t) <= 0) or np.max(np.diff(t)) > cfg['max_sync_gap_s']):
        raise ValueError('Incomplete IMU coverage')
    # Split zero crossings before integrating the magnitude of the interpolated rate.
    i = np.flatnonzero(rate[:-1]*rate[1:] < 0)
    zeros = t[i] - rate[i]*(t[i+1]-t[i])/(rate[i+1]-rate[i])
    knots = np.unique(np.r_[t, zeros])
    speed = np.abs(np.interp(knots, t, rate))
    integral = np.r_[0., np.cumsum(np.diff(knots)*(speed[:-1]+speed[1:])/2)]

    def area(x):
        j = np.clip(knots.searchsorted(x, side='right')-1, 0, len(knots)-2)
        return integral[j] + (x-knots[j])*(speed[j]+np.interp(x, knots, speed))/2

    # For A(x)=integral from x to x+1, A'(x)=speed(x+1)-speed(x).
    edges = np.unique(np.clip(np.r_[0., knots, knots-1., span-1.], 0., span-1.))
    derivative = np.interp(edges+1., knots, speed) - np.interp(edges, knots, speed)
    j = np.flatnonzero(derivative[:-1]*derivative[1:] < 0)
    roots = edges[j] - derivative[j]*np.diff(edges)[j]/np.diff(derivative)[j]
    candidates = np.unique(np.r_[edges, roots])
    areas = area(candidates+1.) - area(candidates)
    best = int(areas.argmax())
    return start+candidates[best]+.5, float(np.degrees(areas[best]))


def experiment(data, solution, out, fastest=False):
    cfg = load_config(data/'profile.json')
    summary = json.loads((solution/'inspection/summary.json').read_bytes())
    if summary['initialization'] != 'delayed' or cfg.get('header_offset_s', {}).get('imu', 0) != 0:
        raise ValueError('Expected blind delayed start and unshifted HEAP-1 gyro clock')
    source = data/'inspection'
    clean = list(scans(source, cfg))
    if len(clean) != 60 or any(s.gyro_delta is None for s in clean[1:]):
        raise ValueError('Expected the 60-scan turn with full IMU coverage')
    t0 = clean[0].t
    center = (clean[0].t + clean[-1].t) / 2
    out.mkdir(parents=True)
    if fastest:
        center, degrees = fastest_second(source, cfg, t0, clean[-1].t)
        (out/'selection.json').write_text(json.dumps(dict(center_s=center-t0, integral_abs_deg=degrees)))
    report = {}
    for label, duration in [('control', 0.), ('half', .5), ('one', 1.), ('repeat', 0.)]:
        begin, end = center - duration/2, center + duration/2
        bag, changed = source, 0
        if duration:
            bag = out/f'bag-{label}'
            changed = mask_rates(source, bag, cfg['imu'], begin, end)
        stream = list(scans(bag, cfg))
        route = out/label
        subprocess.run([sys.executable, '-m', 'mgts_nav', 'localize', str(bag), '--config',
                        str(data/'profile.json'), '--map', str(solution/'map/map.yaml'),
                        '--initial', *map(str, summary['initial_pose']), '--out', str(route)], check=True)
        with (route/'trajectory.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        if [float(r['t']) for r in rows] != [s.t for s in clean]:
            raise ValueError('Scan timestamps differ')
        missing = [s.t-t0 for s in stream[1:] if s.gyro_delta is None]
        after = [(s, r) for s, r in zip(stream, rows, strict=True) if s.t >= end]
        returning = next((s.t-end for s, r in after if s.gyro_delta is not None), None)
        lost = any(r['status'] != 'matched' for s, r in zip(stream, rows) if s.t >= begin)
        matched_again = next((s.t-end for s, r in after if r['status'] == 'matched'), None)
        report[label] = dict(start_s=begin-t0, end_s=end-t0, masked_messages=changed,
                             unavailable_endpoints_s=missing, matched=sum(r['status']=='matched' for r in rows),
                             gyro_delay_s=returning if duration else None, lost_match=lost,
                             first_post_gap_match_delay_s=matched_again if duration else None)
    if (out/'control/trajectory.csv').read_bytes() != (out/'repeat/trajectory.csv').read_bytes():
        raise ValueError('Clean replay differs from first control')
    truth = read_csv(data/'inspection-reference.csv')
    for label, row in report.items():
        est = read_csv(out/label/'trajectory.csv')
        if len(truth) != len(est) or not np.allclose([p.t for p in truth], [p.t for p in est],
                                                      atol=1e-6, rtol=0):
            raise ValueError('Reference timestamps differ')
        metrics = evaluate(est, truth)
        if metrics['coverage'] != 1.:
            raise ValueError('Incomplete CPT7 coverage')
        row.update(ate_m=metrics['ate_xy_rmse_m'], yaw_rmse_deg=np.degrees(metrics['absolute_yaw_rmse_rad']))
        angles = np.array([p.yaw for p in est]) - [p.yaw for p in truth]
        row['max_abs_yaw_error_deg'] = float(np.degrees(np.abs(np.arctan2(np.sin(angles), np.cos(angles)))).max())
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'solution', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--fastest', action='store_true', help='Center outages on the fastest IMU second')
    args = parser.parse_args()
    experiment(args.data, args.solution, args.output, args.fastest)
