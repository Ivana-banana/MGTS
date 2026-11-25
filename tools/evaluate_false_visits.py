"""Count wholly false office zone visits outside the analytic inspection path."""
import argparse
import json
from pathlib import Path

from mgts_nav.bag import load_config, scans
from mgts_nav.demo import PROFILE
from mgts_nav.grid import read_map
from mgts_nav.metrics import read_csv
from mgts_nav.slam import track
from mgts_nav.zones import inside, visits
from tools.evaluate_motion_disagreement import perturb


def negative_zones():
    # The continuous office inspection reaches all four sides of this bounding box.
    xmin, xmax, ymin, ymax = -1., 2., .55, 1.30
    zones = {}
    for cm in (5, 20, 50):
        d = cm / 100
        limits = {
            "west": (xmin-d-.10, ymin-.25, xmin-d, ymax+.25),
            "east": (xmax+d, ymin-.25, xmax+d+.10, ymax+.25),
            "south": (xmin-.25, ymin-d-.10, xmax+.25, ymin-d),
            "north": (xmin-.25, ymax+d, xmax+.25, ymax+d+.10),
        }
        for side, (x0, y0, x1, y1) in limits.items():
            zones[f"{side}-{cm}"] = [[x0,y0], [x1,y0], [x1,y1], [x0,y1]]
    return zones


def union_duration(intervals):
    end = float("-inf")
    total = 0.
    for left, right in sorted((v[1], v[2]) for v in intervals):
        total += max(0., right - max(left, end))
        end = max(end, right)
    return total


def experiment(root):
    zones = negative_zones()
    results = []
    for seed in (17, 101, 202, 303, 404):
        directory = root / f"office-{seed}"
        data = directory / "data"
        cloud = read_map(directory / "map.yaml")[2]
        samples = list(scans(data / "inspection", load_config(PROFILE)))
        truth = read_csv(data / "inspection_truth.csv")
        if any(abs(s.t-p.t) > 1e-6 for s, p in zip(samples, truth, strict=True)):
            raise ValueError("Reference timestamps differ")
        if any(inside(p.x, p.y, polygon) for p in truth for polygon in zones.values()):
            raise ValueError("A reference sample lies in a negative control")
        start = json.loads((data / "initial_pose.json").read_bytes())
        for case in ("clean", "range3"):
            rows = [dict(t=r.t, x=r.pose[0], y=r.pose[1], status=r.status)
                    for r in track(perturb(samples, case), start, map_points=cloud)]
            if [r["t"] for r in rows] != [s.t for s in samples]:
                raise ValueError("Incomplete trajectory")
            found = visits(rows, zones, .25)
            results.append(dict(seed=seed, case=case, scans=len(rows), visits=found,
                                seconds=sum(v[2]-v[1] for v in found), union_s=union_duration(found)))
    return dict(zones=zones, trials=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path, help="Office scenes from experiment 47")
    args = parser.parse_args()
    print(json.dumps(experiment(args.experiment), indent=2, allow_nan=False))
