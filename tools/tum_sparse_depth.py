"""Simulate co-located sparse angular returns on the fixed Freiburg 1 RGB-D pairs."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from tools.tum_depth_motion import run, score
from tools.tum_visual_yaw import K


TOLERANCE = np.array([.15, .25])  # azimuth and elevation, degrees


class BeamGrid:
    def __init__(self, channels):
        az, el = np.meshgrid(np.arange(-40., 40.125, .25), np.linspace(-15., 15., channels))
        angles = np.column_stack((az.ravel(), el.ravel()))
        a, e = np.radians(angles).T
        x, y = np.tan(a), -np.tan(e)/np.cos(a)
        # Forward distortion maps a continuous beam direction to one raw registered depth pixel.
        r2 = x*x + y*y
        radial = 1+r2*(.2624+r2*(-.9531+r2*1.1633))
        xd = x*radial - .0108*x*y + .0026*(r2+2*x*x)
        yd = y*radial - .0054*(r2+2*y*y) + .0052*x*y
        u = np.rint(K[0, 0]*xd+K[0, 2]).astype(int)
        v = np.rint(K[1, 1]*yd+K[1, 2]).astype(int)
        ok = (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
        self.u, self.v = u[ok], v[ok]
        self.rays = np.column_stack((x[ok], y[ok], np.ones(ok.sum())))
        self.angles = angles[ok]

    def observe(self, depth):
        z = depth[self.v, self.u] / 5000.
        valid = z > 0
        return Returns(self.angles[valid], self.rays[valid]*z[valid, None])

    def pairs(self, matches, images, coords):
        views = [self.observe(image) for image in images]
        clouds = [[], []]
        used = [set(), set()]
        for _, u0, v0, u1, v1 in matches:
            ids = [views[0].find(u0, v0), views[1].find(u1, v1)]
            if any(i is None or i in seen for i, seen in zip(ids, used)):
                continue
            for cloud, seen, view, i in zip(clouds, used, views, ids):
                cloud.append(view.points[i])
                seen.add(i)
        return clouds


class Returns:
    def __init__(self, angles, points):
        self.points = points
        self.tree = cKDTree(angles/TOLERANCE)

    def find(self, u, v):
        x, y = (u-K[0, 2])/K[0, 0], (v-K[1, 2])/K[1, 1]
        local = np.degrees([np.arctan(x), np.arctan2(-y, np.hypot(x, 1))])/TOLERANCE
        candidates = self.tree.query_ball_point(local, 1., p=np.inf)
        if candidates:
            return min(candidates, key=lambda i: (np.linalg.norm(self.tree.data[i]-local), i))
        return None


def experiment(data, output, previous):
    output.mkdir(parents=True, exist_ok=False)
    report = {}
    for mode in ('dense', 16, 32):
        rows = run(data) if mode == 'dense' else run(data, BeamGrid(mode).pairs)
        scores = score(rows, data/'groundtruth.txt')
        result = dict(summary=scores, pairs=rows)
        payload = json.dumps(result, indent=2, allow_nan=False)
        if mode == 'dense' and previous is not None and json.loads(payload) != json.loads(previous.read_bytes()):
            raise ValueError('Dense control differs from #156')
        (output/f'{mode}.json').write_text(payload, encoding='utf-8')
        for gap, metrics in scores.items():
            n = [r.get('depth_tracks', 0) for r in rows if r['gap'] == gap]
            metrics.update(total=len(n), depth_tracks=sum(n), median=float(np.median(n)),
                           enough=sum(count >= 8 for count in n))
        report[mode] = scores
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sequence', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--dense-reference', type=Path)
    args = parser.parse_args()
    experiment(args.sequence, args.output, args.dense_reference)
