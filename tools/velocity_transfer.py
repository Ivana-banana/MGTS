"""Freeze and compare causal velocity-prior policies on common scan streams (#163)."""
import argparse
from collections import deque
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from mgts_nav.bag import Scan, load_config, scans
from mgts_nav.coarse_start import estimate
from mgts_nav.demo import lidar_points
from mgts_nav.grid import Grid, read_map
from mgts_nav.metrics import read_csv
from mgts_nav.native_grid import NativeGrid, load_dll
from mgts_nav.slam import VelocityPrior, predict, relative, track
from tools.kitti_repeat import calibration, load_scan, report


class ConsistentPrior(VelocityPrior):
    def __init__(self):
        super().__init__()
        self.recent = deque(maxlen=3)

    def observe(self, row):
        super().observe(row)
        if row.status != "matched" or row.inliers < .55:
            self.recent.clear()
        else:
            self.recent.append(row)

    def predict(self, pose, previous, scan):
        if len(self.recent) == 3:
            a, b, c = self.recent
            dt = np.diff([a.t, b.t, c.t, scan.t])
            if np.all((dt > 0) & (dt <= .2)):
                v0 = relative(a.pose, b.pose)[:2] / dt[0]
                v1 = relative(b.pose, c.pose)[:2] / dt[1]
                if np.linalg.norm(v1-v0) <= .3:
                    return super().predict(pose, previous, scan)
        return predict(pose, previous, scan)


def synth(path):
    rng = np.random.default_rng(163)
    profile = {"base_from_lidar_xyz_m": [0, 0, 0], "base_from_lidar_xyzw": [0, 0, 0, 1]}
    times, poses = [], []
    for i in range(50):
        if path == "gap" and 18 <= i < 24:
            continue
        u = np.clip((i-8)*.07, 0, 2)
        p = {"start_stop": [u, 0, 0], "lateral": [0, u, 0],
             "turn": [.5*np.sin(u), .5*(1-np.cos(u)), u], "gap": [u, 0, 0]}[path]
        times.append(i/10)
        poses.append(p)
    poses = np.array(poses)
    def render(times, poses):
        return [Scan(t, lidar_points(p, t, profile, rng)[:, :2], np.zeros(2), None, None)
                for t, p in zip(times, poses, strict=True)]
    return render(times, poses), poses, [("return", render(times, poses[::-1]), poses[::-1])]


def heap(directory, heldout):
    def read(base, run):
        stream = list(scans(base/run, load_config(base/"profile.json")))
        poses = read_csv(base/f"{run}-reference.csv")
        np.testing.assert_allclose([s.t for s in stream], [p.t for p in poses], rtol=0, atol=1e-6)
        return stream, np.array(poses)[:, 1:]
    windows = []
    for first, base in ((2560, directory), (2620, heldout)):
        stream, truth = read(base, "inspection")
        for i in range(0, len(stream), 60):
            windows.append((str(first+i), stream[i:i+60], truth[i:i+60]))
    return (*read(directory, "mapping"), windows)


def kitti(directory, field):
    with ZipFile(directory/"drive.zip") as archive, ZipFile(directory/"calib.zip") as calib:
        transform = calibration(calib)
        first = json.loads((field/"scores.json").read_text())["inspection_first"]
        def read(name, offset):
            rows = read_csv(field/f"{name}-reference.csv")
            return [load_scan(archive, offset+i, p.t, *transform) for i, p in enumerate(rows)], np.array(rows)[:, 1:]
        return (*read("mapping", 0), [("repeat", *read("inspection", first))])


def scenario(directory, inputs, resolution, dll):
    mapping, truth, windows = inputs
    result = {}
    for method, prior in (("off", None), ("stock", VelocityPrior()), ("gated", ConsistentPrior())):
        dest = directory/method
        dest.mkdir(parents=True, exist_ok=False)
        grid = NativeGrid(resolution, dll) if dll else Grid(resolution)
        rows = list(track(mapping, np.zeros(3), grid=grid, velocity_prior=prior))
        grid.save(dest)
        pixels, meta, points = read_map(dest/"map.yaml")
        mapping_score = report([s.t for s in mapping], truth, [r.pose for r in rows], [r.status for r in rows])
        result[method] = {"mapping": mapping_score, "applied": prior.applied if prior else 0, "frames": len(rows)}
        for name, stream, reference in windows:
            try:
                start = estimate(stream[0], pixels, meta, {})
                diff = relative(reference[0], start)
                blind = "correct" if np.linalg.norm(diff[:2]) <= .25 and abs(diff[2]) <= np.deg2rad(10) else "false_accept"
            except ValueError as exc:
                blind = str(exc)
            rows = list(track(stream, reference[0], map_points=points))
            result[method][name] = dict(blind=blind, **report([s.t for s in stream], reference,
                                          [r.pose for r in rows], [r.status for r in rows]))
        (dest/"scores.json").write_text(json.dumps(result[method], indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dll", type=Path)
    args = parser.parse_args()
    dll = load_dll(args.dll) if args.dll else None
    for name in ("start_stop", "lateral", "turn", "gap", "kitti", "heap"):
        if name == "kitti":
            data = kitti(Path("out/kitti161"), Path("out/kitti161-run"))
        elif name == "heap":
            data = heap(Path("out/heap1-long-80"), Path("out/heap1-heldout-37"))
        else:
            data = synth(name)
        scenario(args.output/name, data, .12 if name == "kitti" else .08, dll)

