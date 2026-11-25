"""Exercise drift recovery after an injected pose jump at the fourth scan."""
import json
import numpy as np

from mgts_nav.bag import Scan
from mgts_nav.recovery import DriftRecovery, Recovery
from mgts_nav.slam import rotation, track


def run(case, experimental):
    room = np.full((41,61),254,dtype=np.uint8)
    room[[0,-1],:] = room[:,[0,-1]] = 0
    room[8:21,12] = room[20,12:28] = 0
    duplicate = room.copy()
    if case != 'symmetric':
        duplicate[1:,:30] = 254
        duplicate[0,:30] = 254
    left = duplicate if case == 'ambiguous_alert' else room
    pixels = np.concatenate((left,np.full((41,25),205,dtype=np.uint8),duplicate),axis=1)
    meta = dict(origin=[-2.,-2.,0.],resolution=.1,occupied_thresh=.65,free_thresh=.196)
    local = (np.argwhere(room[::-1]==0)[:,::-1]+.5)*.1-2
    points = (np.argwhere(pixels[::-1]==0)[:,::-1]+.5)*.1-2
    truth = [np.array([.50+.04*i,-.27,.38]) for i in range(12)]
    stream = [Scan(i*.1,(local-p[:2])@rotation(p[2]),np.zeros(2),p,None) for i,p in enumerate(truth)]
    if case == 'foreign':
        for s in stream[3:]:
            s.xy *= 1.8
    if case == 'outage':
        for s in stream[3:6]:
            s.xy = np.empty((0,2))
    recovery = Recovery(pixels,meta)
    alternate = DriftRecovery(pixels,meta) if experimental else None
    def segment(rows,start):
        return list(track(rows,start,map_points=points,recovery=recovery,drift_recovery=alternate))
    prefix = segment(stream[:3],truth[0])
    shift = 0 if case in ('correct','outage') else 8.6
    rows = prefix+segment(stream[3:],truth[3]+[shift,0,0])
    large = [np.linalg.norm(r.pose[:2]-p[:2])>.2 for r,p in zip(rows,truth)]

    restored = next((r.t-stream[3].t for r,p in zip(rows[3:],truth[3:]) if
                     r.status=='matched' and np.linalg.norm(r.pose[:2]-p[:2])<=.2),None)
    events = alternate.stats['history'] if alternate else []
    bad_switches = sum(np.linalg.norm(np.array(e['pose'][:2])-truth[round(e['t']*10)][:2])>.2
                       for e in events if e['action']=='confirmed')
    return dict(statuses=[r.status for r in rows],wrong_matched_s=.1*sum(
        v and r.status=='matched' for v,r in zip(large,rows)),restore_s=restored,
        false_switches=int(bad_switches),history=events)


if __name__ == '__main__':
    print(json.dumps({case:[run(case,False),run(case,True)] for case in
                      ('correct','jump','symmetric','ambiguous_alert','foreign','outage')},indent=2))
