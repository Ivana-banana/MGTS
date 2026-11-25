"""Ablate free-space evidence and deferred choice among global scan modes."""
import argparse
from dataclasses import replace
import json
from itertools import islice
from pathlib import Path
from time import process_time

import numpy as np
from scipy.spatial import cKDTree

from mgts_nav.bag import load_config, scans
from mgts_nav.coarse_start import estimate
from mgts_nav.delayed_start import compose, estimate_prepared, prepare_window
from mgts_nav.grid import read_map
from mgts_nav.metrics import read_csv, wrap
from mgts_nav.slam import world


def choose(window, pixels, meta, *, rays, delayed):
    modes = []
    try:
        estimate(window.rows[0], pixels, meta, {}, candidates=modes)
    except ValueError:
        if not modes:
            raise
    if delayed and window.rejection:
        raise ValueError(window.rejection)
    occupancy = 1 - np.flipud(pixels).astype(float)/255
    occupied = occupancy > meta['occupied_thresh']
    free = occupancy < meta['free_thresh']
    cell = meta['resolution']
    origin = np.array(meta['origin'][:2])
    tree = cKDTree(origin+(np.argwhere(occupied)[:,::-1]+.5)*cell)
    scores = []
    for mode in modes:
        evidence = []
        for i in ([0,10,20] if delayed else [0]):
            scan = window.rows[i]
            pose = compose(mode,window.local[i]) if delayed else mode
            ends = world(scan.xy,pose)
            distance = tree.query(ends)[0]
            endpoint = np.exp(-.5*(distance/.10)**2).mean()
            if rays:
                start = world(scan.origin,pose)
                samples = start + np.array([.2,.4,.6,.8])[:,None,None]*(ends-start)
                x,y = np.floor((samples-origin)/cell).astype(int).reshape(-1,2).T
                inside = (x>=0)&(y>=0)&(x<pixels.shape[1])&(y<pixels.shape[0])
                values = np.full(len(x),.5)
                ix,iy = x[inside],y[inside]
                values[inside] = .5 + .5*free[iy,ix] - .5*occupied[iy,ix]
                endpoint = .8*endpoint + .2*values.mean()
            evidence.append(endpoint)
        scores.append(float(np.mean(evidence)))
    order = np.argsort(scores)[::-1]
    if scores[order[0]] < .55 or (len(order)>1 and scores[order[0]]-scores[order[1]]<=.045):
        raise ValueError('Weak or ambiguous')
    return modes[order[0]]


def sections(window, pixels, meta):
    # Equal spatial halves on the long map axis, each with a one-metre overlap.
    axis = int(pixels.shape[1] > pixels.shape[0])
    length = pixels.shape[axis]
    margin = round(1/meta['resolution'])
    accepted = []
    for start,end in ((0,min(length,length//2+margin)),(max(0,length//2-margin),length)):
        mask = pixels.copy()
        outside = np.ones(length,dtype=bool)
        outside[start:end] = False
        if axis:
            mask[:,outside] = 205
        else:
            mask[outside,:] = 205
        try:
            accepted.append(estimate_prepared(window,mask,meta,{}))
        except ValueError:
            pass
    if len(accepted)!=1:
        raise ValueError(f'{len(accepted)} accepted sections')
    return accepted[0]


def run(args):
    source = scans(args.bag,load_config(args.profile))
    rows = list(islice(source,args.skip,args.skip+21))
    if args.range_m:
        rows = [replace(s,xy=s.xy[np.linalg.norm(s.xy-s.origin,axis=1)<=args.range_m]) for s in rows]
    truth = read_csv(args.truth)[args.skip]
    if abs(truth.t-rows[0].t)>1e-6:
        raise ValueError('Reference timestamp mismatch')
    pixels, meta, _ = read_map(args.map)
    if args.double:
        pixels = np.concatenate((pixels,np.full((len(pixels),25),205,dtype=pixels.dtype),pixels),axis=1)
    result = {}
    for method in ('delayed','sections','endpoints','rays','sequence','sequence_rays'):
        start = process_time()
        try:
            window = prepare_window(rows if method not in ('endpoints','rays') else rows[:1])
            if method == 'delayed':
                pose = estimate_prepared(window,pixels,meta,{})
            elif method == 'sections':
                pose = sections(window,pixels,meta)
            else:
                pose = choose(window,pixels,meta,rays='rays' in method,delayed='sequence' in method)
        except ValueError as exc:
            value = dict(status='refused',reason=str(exc))
        else:
            xy = np.linalg.norm(pose[:2]-truth[1:3])
            yaw = np.rad2deg(abs(wrap(pose[2]-truth.yaw)))
            good = not args.foreign and xy<=.5 and yaw<=12
            value = dict(status='correct' if good else 'false',xy_m=xy,yaw_deg=yaw)
        index = 0 if method in ('endpoints','rays') else 20
        result[method] = dict(**value,cpu_s=process_time()-start,horizon_s=rows[index].t-rows[0].t)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bag','profile','map','truth'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--range-m',type=float)
    parser.add_argument('--skip',type=int,default=0)
    parser.add_argument('--double',action='store_true')
    parser.add_argument('--foreign',action='store_true')
    print(json.dumps(run(parser.parse_args()),indent=2))
