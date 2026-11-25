"""Report sensitivity of zone visits and obstacle detections to alternative map poses."""
import argparse
import json
from pathlib import Path

import numpy as np

from .bag import load_config, scans
from .grid import read_map
from .obstacles import detect, matched_poses
from .zones import inside, load_zones


def assess(stream, poses, pixels, meta, zones, offsets, tolerance=.25):
    offsets = np.asarray(offsets, dtype=float)
    if offsets.ndim != 2 or offsets.shape[1] != 3 or not np.isfinite(offsets).all():
        raise ValueError('Expected finite map-frame dx, dy, dyaw triples')
    variants = [poses] + [{t: p+shift for t,p in poses.items()} for shift in offsets]
    zone_rows = []
    for scan in stream:
        states = {}
        for name, polygon in zones.items():
            if scan.t not in poses or not len(scan.xy):
                state = 'unobserved'
            else:
                flags = [inside(*v[scan.t][:2],polygon) for v in variants]
                state = 'confirmed' if all(flags) else 'uncertain' if any(flags) else 'outside'
            states[name] = state
        zone_rows.append(dict(t=scan.t,zones=states))

    intervals = {}
    for name in zones:
        runs = []
        for sample in zone_rows:
            state = sample['zones'][name]
            if runs and runs[-1]['state'] == state and sample['t']-runs[-1]['last_s'] <= .5:
                runs[-1]['last_s'] = sample['t']
            else:
                runs.append(dict(state=state, first_s=sample['t'], last_s=sample['t']))
        intervals[name] = runs
    clusters = []
    for i, variant in enumerate(variants):
        events = detect(stream,variant,pixels,meta)['events']
        # One event per pose hypothesis per cluster, closest compatible position first.
        pairs = []
        for j,event in enumerate(events):
            for k,cluster in enumerate(clusters):
                reference = cluster[0][1]
                if max(reference['first_s'],event['first_s']) <= min(reference['last_s'],event['last_s']):
                    d = np.hypot(reference['x_m']-event['x_m'],reference['y_m']-event['y_m'])
                    if d <= tolerance:
                        pairs.append((d,j,k))
        used, assigned = set(),set()
        for _,j,k in sorted(pairs):
            if j not in used and k not in assigned:
                clusters[k].append((i,events[j]))
                used.add(j)
                assigned.add(k)
        clusters.extend([[(i,e)] for j,e in enumerate(events) if j not in used])
    observations = []
    for cluster in clusters:
        positions = np.array([[e['x_m'],e['y_m']] for _,e in cluster])
        span = np.linalg.norm(positions[:,None]-positions,axis=2).max()
        common_start = max(e['confirmed_s'] for _,e in cluster)
        common_end = min(e['last_s'] for _,e in cluster)
        confirmed = len(cluster)==len(variants) and span<=tolerance and common_start<=common_end
        observations.append(dict(status='confirmed' if confirmed else 'uncertain',
                                 hypothesis_ids=[i for i,_ in cluster], first_s=min(e['first_s'] for _,e in cluster),
                                 confirmed_s=common_start if confirmed else None, location_span_m=span,
                                 x_m=positions[:,0].mean(), y_m=positions[:,1].mean()))
    observed = [s.t for s in stream if s.t in poses and len(s.xy)]
    return dict(map_frame_offsets=offsets.tolist(), obstacle_tolerance_m=tolerance,zones=zone_rows, zone_intervals=intervals,
                unobserved_scans=[s.t for s in stream if s.t not in poses or not len(s.xy)],
                obstacle_status=('unobserved' if not observed else 'absent' if not observations else 'events'),
                obstacles=observations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bag','trajectory','map','profile','zones','out'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--offset',action='append',nargs=3,type=float,required=True,metavar=('DX','DY','DYAW'))
    args = parser.parse_args()
    pixels,meta,_ = read_map(args.map)
    result = assess(list(scans(args.bag,load_config(args.profile))),matched_poses(args.trajectory),
                    pixels,meta,load_zones(args.zones),args.offset)
    with args.out.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
