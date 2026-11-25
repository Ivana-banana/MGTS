"""Paired office evaluation of pose-sensitive event reports from issue 154."""
import argparse
import json
from pathlib import Path

import numpy as np

from mgts_nav.bag import load_config,scans
from mgts_nav.demo import PROFILE
from mgts_nav.event_uncertainty import assess
from mgts_nav.grid import read_map
from mgts_nav.metrics import read_csv
from mgts_nav.obstacles import matched_poses
from mgts_nav.zones import load_zones,inside
from tools.obstacle_pose_controls import TARGET


def run(root):
    pixels,meta,_ = read_map(root/'map/map.yaml')
    zones = load_zones(PROFILE.parent/'demo-zones.json')
    result = {}
    for case in sorted(root.glob('*-*-*')):
        stream = list(scans(case/'inspection',load_config(PROFILE)))
        poses = matched_poses(case/'localized/trajectory.csv')
        truth = read_csv(case/'inspection_truth.csv')
        obj = case.name.endswith('object')
        for label,offsets in [('nominal',[[0,0,0]]),('alternatives',[[-.45,0,0],[.45,0,0]])]:
            report = assess(stream,poses,pixels,meta,zones,offsets)
            tp=fp=fn=uncertain=0
            for sample,p in zip(report['zones'],truth,strict=True):
                assert abs(sample['t']-p.t)<1e-6
                for name,state in sample['zones'].items():
                    actual = inside(p.x,p.y,zones[name])
                    tp += actual and state=='confirmed'
                    fp += not actual and state=='confirmed'
                    fn += actual and state!='confirmed'
                    uncertain += state=='uncertain'
            zvis_tp=zvis_fp=zvis_fn=0
            zdelays=[]
            for name,polygon in zones.items():
                target=[]
                last_inside=False
                for p in truth:
                    active=inside(p.x,p.y,polygon)
                    if active and not last_inside:
                        target.append([p.t,p.t])
                    if active:
                        target[-1][1]=p.t
                    last_inside=active
                matched=set()
                for visit in report['zone_intervals'][name]:
                    if visit['state']!='confirmed':
                        continue
                    idx=next((i for i,(a,b) in enumerate(target) if i not in matched and
                              a<=visit['last_s'] and visit['first_s']<=b),None)
                    if idx is None:
                        zvis_fp+=1
                    else:
                        zvis_tp+=1
                        matched.add(idx)
                        zdelays.append(visit['first_s']-target[idx][0])
                zvis_fn+=len(target)-len(matched)
            events = report['obstacles']
            accepted = [e for e in events if e['status']=='confirmed']
            onset = 252.
            matches = [(e['confirmed_s']-onset,np.hypot(e['x_m']-TARGET[0],e['y_m']-TARGET[1]))
                       for e in accepted if obj and e['confirmed_s']>=onset and
                       np.hypot(e['x_m']-TARGET[0],e['y_m']-TARGET[1])<=.45]
            best = min(matches) if matches else (None,None)
            result[f'{case.name}/{label}'] = dict(zone_tp=tp,zone_fp=fp,zone_fn=fn,zone_uncertain=uncertain,
                zone_slots=len(truth)*len(zones),visit_tp=zvis_tp,visit_fp=zvis_fp,visit_fn=zvis_fn,visit_delays=zdelays,obstacle_tp=int(bool(matches)),obstacle_fp=len(accepted)-bool(matches),
                obstacle_fn=int(obj and not matches),obstacle_uncertain=len(events)-len(accepted),
                obstacle_events=len(events),delay_s=best[0],position_error_m=best[1])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paired',type=Path)
    print(json.dumps(run(parser.parse_args().paired),indent=2))
