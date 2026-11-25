"""Evaluate windowed clock diagnostics on injected faults and HEAP-1."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from mgts_nav.bag import imu_values, load_config, reader, scans, stamp
from mgts_nav.clock_windows import diagnose, interpolate
from mgts_nav.slam import match


def rate(t):
    motion = .48*np.sin(1.7*t)+.23*np.sin(3.1*t+.4)+.04*np.sin(.3*t)
    return np.where((t >= 20) & (t < 30), 0., motion)


def save(path, t, v):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["t","rate"])
        writer.writerows(zip(t, v, strict=True))


def trial(output, correct):
    output.mkdir()
    t = np.arange(6001)/100
    # Angle and body speed define the independent kinematics truth trajectory.
    yaw = np.r_[0,np.cumsum((rate(t[1:])+rate(t[:-1]))*.005)]
    velocity = (.6+.2*np.sin(t))[:,None]*np.column_stack((np.cos(yaw),np.sin(yaw)))
    velocity[(t >= 20) & (t < 30)] = 0.
    xy = np.vstack((np.zeros(2),np.cumsum((velocity[1:]+velocity[:-1])*.005,axis=0)))
    ref = t[::10]
    # LiDAR orientation observations, differentiated over each consecutive pair.
    midpoint = (ref[1:]+ref[:-1])/2
    lidar = np.diff(yaw[::10])/.1
    # Extend by one real measurement at each endpoint (one-sided difference).
    reference = (np.r_[ref[0],midpoint,ref[-1]], np.r_[lidar[0],lidar,lidar[-1]])
    save(output/"lidar.csv", *reference)
    scores = {}
    for name in ("constant","drift","jump"):
        offset = .07 + {"constant":np.zeros_like(t), "drift":.003*t, "jump":.18*(t>=40)}[name]
        for sensor in ("imu","odom"):
            sampling = 2 if sensor == "imu" else 5
            ids = np.arange(0,len(t),sampling)
            ids = ids[(t[ids] < 10) | (t[ids] >= 15)]
            ts = t[ids]+offset[ids]
            measured = rate(t[ids]) if sensor == "imu" else np.gradient(yaw,t)[ids]
            save(output/f"{name}-{sensor}.csv", ts, measured)
            result = diagnose(reference,(ts,measured))
            usable = [r for r in result["windows"] if r["lag_s"] is not None]
            center = np.array([(r["start_s"]+r["end_s"])/2 for r in usable])
            estimated = np.array([r["lag_s"] for r in usable])
            result["max_lag_error_s"] = float(max(abs(estimated-np.interp(center,t,offset))))
            result["false_jumps"] = sum(name != "jump" or abs(v-40)>2.5 for v in result["jumps_s"])
            # Position sampling checks timestamp alignment of the odometry itself.
            if sensor == "odom":
                for mode in (("raw","corrected") if correct else ("raw",)):
                    errors = []
                    for row in usable:
                        query = ref[(ref >= row["start_s"]+.6) & (ref < row["end_s"]-.6)]
                        lag = row["lag_s"] if mode == "corrected" else 0
                        pred = np.column_stack([interpolate(ts,col[ids],query+lag) for col in xy.T])
                        target = np.column_stack([np.interp(query,t,col) for col in xy.T])
                        errors.extend(np.linalg.norm(pred-target,axis=1))
                    result[mode+"_ate_m"] = float(np.sqrt(np.nanmean(np.square(errors))))
            scores[name+"-"+sensor] = result
    return scores


def heap(data, output):
    cfg = load_config(data/"profile.json")
    stream = list(scans(data/"mapping",dict(cfg,imu=None,odom=None)))
    ts, rates = [], []
    for a,b in zip(stream,stream[1:]):
        if not len(a.xy):
            continue
        pose, _, ok = match(b.xy,cKDTree(a.xy),np.zeros(3))
        if ok:
            ts.append((a.t+b.t)/2)
            rates.append(pose[2]/(b.t-a.t))
    sample = []
    with reader(data/"mapping") as bag:
        rot = Rotation.from_quat(cfg["base_from_imu_xyzw"])
        for c,_,raw in bag.messages(connections=[c for c in bag.connections if c.topic == cfg["imu"]]):
            msg = bag.deserialize(raw,c.msgtype)
            value, valid, _ = imu_values(msg,rot,False)
            if valid:
                sample.append((stamp(msg),value[2]))
    reference, sensor = (np.array(ts), np.array(rates)), np.array(sample).T
    save(output/"heap-lidar.csv",*reference)
    save(output/"heap-imu.csv",*sensor)
    return diagnose(reference,sensor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--apply-estimated",action="store_true",help="also score offset-adjusted synthetic odometry")
    parser.add_argument("--heap",type=Path,default=Path("out/heap1-long-80"))
    args = parser.parse_args()
    result = trial(args.output,args.apply_estimated)
    result["heap"] = heap(args.heap,args.output)
    (args.output/"scores.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")

