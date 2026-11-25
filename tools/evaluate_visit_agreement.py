"""Compare visit intervals from localization and anchored wheel odometry (#119)."""
import argparse
import json
from pathlib import Path

import numpy as np

from mgts_nav.bag import load_config, scans
from mgts_nav.demo import PROFILE
from mgts_nav.grid import read_map
from mgts_nav.metrics import read_csv
from mgts_nav.slam import relative, track, world
from mgts_nav.zones import load_zones, visits
from tools.evaluate_false_visits import negative_zones
from tools.evaluate_motion_disagreement import perturb


def overlap(a, b, times):
    if a[0] != b[0]:
        return 0
    left = max(a[1], b[1])
    right = min(a[2], b[2])
    return max(0, int(times.searchsorted(right, side="right") - times.searchsorted(left)))


def confirmed(local, wheel, times):
    return [v for v in local if any(overlap(v, w, times) >= max(2, (v[5]+1)//2) for w in wheel)]


def score(intervals, truth, times):
    true_samples = sum(overlap(v, t, times) for v in intervals for t in truth)
    false = [v for v in intervals if all(overlap(v, t, times) == 0 for t in truth)]
    return dict(found=sum(any(overlap(v, t, times) >= 2 for v in intervals) for t in truth),
                true_samples=true_samples, extra_samples=sum(v[5] for v in intervals)-true_samples,
                false_visits=len(false), false_seconds=sum(v[2]-v[1] for v in false))


def experiment(root):
    zones = {**load_zones(Path("configs/demo-zones.json")), **negative_zones()}
    for seed in (17, 101, 202, 303, 404):
        directory = root / f"office-{seed}"
        data = directory / "data"
        cloud = read_map(directory/"map.yaml")[2]
        original = list(scans(data/"inspection", load_config(PROFILE)))
        start = json.loads((data/"initial_pose.json").read_bytes())
        refs = read_csv(data/"inspection_truth.csv")
        times = np.array([s.t for s in original])
        np.testing.assert_allclose(times, [r.t for r in refs], rtol=0, atol=1e-6)
        truth = visits([dict(t=t,x=r.x,y=r.y,status="matched") for t,r in zip(times,refs,strict=True)], zones, .25)
        for case in ("clean", "range3", "range3-drift"):
            selected = list(perturb(original, "clean" if case == "clean" else "range3"))
            if case == "range3-drift":
                selected = list(perturb(selected, "drift"))
            local_rows = [dict(t=r.t, x=r.pose[0], y=r.pose[1], status=r.status)
                          for r in track(selected, start, map_points=cloud)]
            wheel_rows = []
            for s in selected:
                if s.odom is None:
                    raise ValueError("Expected complete demo odometry")
                x, y = world(relative(selected[0].odom, s.odom)[:2], start)
                wheel_rows.append(dict(t=s.t, x=x, y=y, status="matched"))
            local = visits(local_rows, zones, .25)
            wheel = visits(wheel_rows, zones, .25)
            kept = confirmed(local, wheel, times)
            yield dict(seed=seed, case=case, truth=truth, local=local, wheel=wheel, kept=kept,
                       scores={name:score(rows,truth,times) for name,rows in
                               (("local",local), ("wheel",wheel), ("kept",kept))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    print(json.dumps(list(experiment(args.experiment)), indent=2, allow_nan=False))
