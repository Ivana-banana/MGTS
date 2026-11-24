"""Run a two-sensor dropout and odometry-origin reset through the navigation CLIs."""
import argparse
import csv
import json
from math import degrees, hypot
from pathlib import Path
import subprocess
import sys

from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from mgts_nav.bag import reader
from mgts_nav.demo import PROFILE, START_NS, generate
from mgts_nav.metrics import read_csv, wrap


def altered_bag(source, target, cfg, *, lose_odom, reset):
    store = get_typestore(Stores.ROS2_HUMBLE)
    removed = {'lidar': 0, 'odom': 0}
    offset = None
    with reader(source) as bag, Writer(target, version=8) as writer:
        outputs = {c.id: writer.add_connection(c.topic, c.msgtype, typestore=store)
                   for c in bag.connections}
        for c, rx, raw in bag.messages():
            if c.topic in (cfg['lidar'], cfg['odom']):
                msg = bag.deserialize(raw, c.msgtype)
                stamp = msg.header.stamp
                dt = stamp.sec * 10**9 + stamp.nanosec - START_NS['inspection']
                if c.topic == cfg['lidar'] and 3_000_000_000 <= dt < 5_000_000_000:
                    removed['lidar'] += 1
                    continue
                if c.topic == cfg['odom']:
                    if lose_odom and 3_500_000_000 <= dt < 4_500_000_000:
                        removed['odom'] += 1
                        continue
                    if reset and dt >= 4_500_000_000:
                        p = msg.pose.pose.position
                        if offset is None:
                            offset = p.x, p.y
                        p.x, p.y = p.x - offset[0], p.y - offset[1]
                        raw = store.serialize_cdr(msg, c.msgtype)
            writer.write(outputs[c.id], rx, raw)
    return dict(removed=removed, subtracted_odom_xy_m=offset)


def invoke(module, *args):
    runner = subprocess.run([sys.executable, '-m', module, *map(str, args)], text=True,
                            stdout=subprocess.PIPE, check=True)
    return json.loads(runner.stdout)


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def score(result, clock, truth):
    rows = read_rows(result/'trajectory.csv')
    dense = read_rows(result/'continuous.csv')
    # Score the first scheduled scan after both interruptions, including the clean control.
    first = next(r for r in rows if float(r['t']) >= clock + 5)
    reference = {p.t: p for p in truth}
    def error(row):
        ref = reference[float(row['t'])]
        return dict(xy_m=hypot(float(row['x'])-ref.x, float(row['y'])-ref.y),
                    yaw_deg=abs(degrees(wrap(float(row['yaw'])-ref.yaw))))
    after = [r for r in rows if float(r['t']) >= float(first['t'])]
    first_match = next((r for r in after if r['status'] == 'matched'), None)
    matched = [r for r in after if r['status'] == 'matched']
    return dict(first_return=dict(t=float(first['t']), status=first['status'], **error(first)),
                first_match=(dict(t=float(first_match['t']), **error(first_match)) if first_match else None),
                later_matched=len(matched), later_samples=len(after),
                later_max_matched_error_m=max((error(r)['xy_m'] for r in matched), default=None),
                emitted_during_lidar_loss=sum(clock+3 <= float(r['t']) < clock+5 for r in dense),
                emitted_between_odom_return_and_lidar=sum(clock+4.5 <= float(r['t']) < clock+5 for r in dense),
                last_output_t=float(dense[-1]['t']),
                tail_without_output_s=max(0., float(rows[-1]['t']) - float(dense[-1]['t'])),
                largest_output_gap_s=max(float(b['t'])-float(a['t']) for a,b in zip(dense, dense[1:])))


def experiment(root, seed):
    data = root/'demo'
    generate(data, seed=seed)
    cfg = json.loads(PROFILE.read_text(encoding='utf-8'))
    invoke('mgts_nav', 'map', data/'mapping', '--config', PROFILE, '--out', root/'map')
    truth = read_csv(data/'inspection_truth.csv')
    original = data/'inspection'
    clean = dict(removed=dict(lidar=0, odom=0), subtracted_odom_xy_m=None)
    cases = {'clean': (original, clean)}
    for case in ('lidar_only', 'joint', 'joint_reset'):
        bag = root / (case+'-bag')
        inputs = altered_bag(original, bag, cfg, lose_odom=case != 'lidar_only', reset=case == 'joint_reset')
        cases[case] = bag, inputs
    cases['joint_reset_recover'] = cases['joint_reset']
    measurements = {}
    for case, (bag, inputs) in cases.items():
        result = root/case
        summary = invoke('mgts_nav', 'localize', bag, '--config', PROFILE, '--map', root/'map/map.yaml',
                         '--initial-file', data/'initial_pose.json', '--out', result,
                         *(['--recover'] if case == 'joint_reset_recover' else []))
        dense = invoke('mgts_nav.continuous', bag, result/'trajectory.csv', result/'continuous.csv',
                       '--config', PROFILE)
        measurements[case] = dict(**inputs, status=summary['status'], recovery=summary.get('recovery'),
                                  odom_gaps=dense['odom_gaps_s'],
                                  max_prediction_age_s=dense['max_prediction_age_s'],
                                  **score(result, START_NS['inspection']/1e9, truth))
    return measurements


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {seed: experiment(args.out/str(seed), seed) for seed in (41, 73)}
    (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, allow_nan=False))
