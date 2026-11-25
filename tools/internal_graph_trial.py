"""Run three map graph choices and a disjoint inspection on each fixed route."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
from rosbags.typesys import Stores, get_typestore

from mgts_nav.demo import PROFILE, WALLS, pose_at, record
from mgts_nav.grid import read_map
from mgts_nav.map_eval import segment_distances
from mgts_nav.metrics import evaluate, read_csv


def run(root):
    root.mkdir(exist_ok=False)
    result = {}
    for case in ('eight','open','corridor'):
        directory = root/case
        directory.mkdir()
        layout = 'corridor' if case == 'corridor' else 'office'
        def route(role,t,layout,duration):
            if role == 'inspection' or case == 'corridor':
                return pose_at(role,t,layout,duration)
            a = 2*np.pi*np.clip((t-1)/(duration-2),0,1)
            if case == 'open':
                return np.array([3*a/(2*np.pi),.4*np.sin(a),.2*np.sin(a)])
            return np.array([2.6*np.sin(a),1.1*np.sin(2*a),.2*np.sin(a)])
        for k,role in enumerate(('mapping','inspection')):
            record(directory,role,json.loads(PROFILE.read_bytes()),get_typestore(Stores.ROS2_HUMBLE),
                   41+k,layout=layout,trajectory=route)
        map_truth, inspect_truth = [read_csv(directory/f'{r}_truth.csv') for r in ('mapping','inspection')]
        for name,options in [('plain',[]),('endpoint',['--close-loop']),('internal',['--internal-loops'])]:
            out = directory/name
            def cli(*args):
                subprocess.run([sys.executable,'-m','mgts_nav',*map(str,args)],check=True,stdout=subprocess.DEVNULL)
            start = perf_counter()
            cli('map',directory/'mapping','--config',PROFILE,'--out',out,*options)
            seconds = perf_counter()-start
            cli('localize',directory/'inspection','--config',PROFILE,'--map',out/'map.yaml',
                '--initial',*inspect_truth[0][1:],'--out',out/'loc')
            selected = []
            if name=='internal':
                selected = json.loads((out/'internal-loops.json').read_text())['selected']
            elif name=='endpoint' and json.loads((out/'loop.json').read_text())['loop_supported']:
                selected = [dict(first=0,second=len(map_truth)-1)]
            false_links = sum(np.linalg.norm(np.array(map_truth[e['first']][1:3])-map_truth[e['second']][1:3])>.5
                              for e in selected)
            walls = WALLS
            if layout == 'corridor':
                border = [(-6,-1.3),(-1.6,-1.3),(-1.6,-2.5),(.1,-2.5),(.1,-1.3),
                          (5.8,-1.3),(5.8,2.1),(3.4,2.1),(3.4,3.8),(1.2,3.8),
                          (1.2,2.1),(-6,2.1),(-6,-1.3)]
                walls = np.array([*zip(border,border[1:]),[(-1.6,-1.3),(-.5,-2.35)]])
            error = segment_distances(read_map(out/'map.yaml')[2],walls)
            result[f'{case}/{name}'] = dict(map_s=seconds,links=len(selected),false_links=int(false_links),
                wall_p95_m=float(np.quantile(error,.95)),
                map_ate_m=evaluate(read_csv(out/'trajectory.csv'),map_truth)['ate_xy_rmse_m'],
                inspect_ate_m=evaluate(read_csv(out/'loc/trajectory.csv'),inspect_truth)['ate_xy_rmse_m'])
            (root/'scores.json').write_text(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    print(json.dumps(run(parser.parse_args().out),indent=2))
