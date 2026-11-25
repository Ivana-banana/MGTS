"""Compare map-point models for localization on rotated, shifted and degraded scenes.

The demo layouts put every wall on a grid-cell border, so cell-centre matching
there has a constant half-cell bias that real maps rarely have. This script
reuses the demo recordings but starts the mapping run from a transformed pose
(rotation and shift of the map frame), so the occupancy grid is no longer
aligned with the walls. Truth and inspection start are transformed the same
way. Variants change grid resolution, limit lidar range during inspection, or
add radial lidar noise to both runs.

Each scenario localizes the same inspection with several point models:
``base`` (cell centres, as on main), ``mean3`` (the first version of PR #43:
plain 3x3 neighbourhood mean) and ``current`` (``slam.track`` with
``surface=True``, the wall-line projection).

Run from the MGTS root:
    python tools/surface_scenes.py out/scenes --data out/scenes/data
"""
import argparse
import copy
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mgts_nav import slam  # noqa: E402
from mgts_nav.bag import load_config, scans  # noqa: E402
from mgts_nav.demo import PROFILE, generate  # noqa: E402
from mgts_nav.grid import Grid, read_map  # noqa: E402
from mgts_nav.metrics import Pose, evaluate, read_csv  # noqa: E402

SEEDS = {"office": (17, 101, 202, 303, 404, 505, 606, 707, 808, 909),
         "corridor": (41, 73, 505, 606, 707, 808, 909, 111, 222, 333)}
# name: (rotation deg, shift m, cell m, inspection range m, lidar noise m)
SCENARIOS = {
    "aligned": (0, 0, .08, None, 0),
    "rot17": (17, 0, .08, None, 0),
    "rot33": (33, 0, .08, None, 0),
    "shift3cm": (0, .03, .08, None, 0),
    "cell5_aligned": (0, 0, .05, None, 0),
    "cell5_rot17": (17, 0, .05, None, 0),
    "cell12_aligned": (0, 0, .12, None, 0),
    "cell12_rot17": (17, 0, .12, None, 0),
    "range4_aligned": (0, 0, .08, 4.0, 0),
    "range4_rot17": (17, 0, .08, 4.0, 0),
    "range3_aligned": (0, 0, .08, 3.0, 0),
    "range3_rot17": (17, 0, .08, 3.0, 0),
    "range3_rot33": (33, 0, .08, 3.0, 0),
    "range2_aligned": (0, 0, .08, 2.0, 0),
    "range2_rot17": (17, 0, .08, 2.0, 0),
    "range2_rot33": (33, 0, .08, 2.0, 0),
    "noise5_aligned": (0, 0, .08, None, .05),
    "noise5_rot17": (17, 0, .08, None, .05),
    "noise5_rot33": (33, 0, .08, None, .05),
}


def mean3(points):
    """First PR #43 version: every point becomes the mean of its occupied 3x3 cells."""
    points = np.asarray(points, dtype=float)
    tree = cKDTree(points)
    spacing = np.median(tree.query(points, k=2)[0][:, 1])
    if not np.isfinite(spacing) or spacing <= 0:
        return points
    neighbours = tree.query_ball_point(points, 1.5 * spacing)
    return np.array([points[ids].mean(axis=0) for ids in neighbours])


# method: (map point transform, track(surface=...))
METHODS = {"base": (None, False), "mean3": (mean3, False), "current": (None, True)}


def transform(pose, yaw, shift):
    c, s = np.cos(yaw), np.sin(yaw)
    x, y = pose[0], pose[1]
    return np.array([c * x - s * y + shift, s * x + c * y + shift, pose[2] + yaw])


def degraded(stream, max_range=None, noise=0.0, rng=None):
    for scan in stream:
        scan = copy.copy(scan)
        xy = scan.xy
        if noise:
            ray = xy - scan.origin
            length = np.linalg.norm(ray, axis=1, keepdims=True)
            extra = rng.normal(0, noise, length.shape)
            xy = scan.origin + ray * (1 + extra / np.maximum(length, 1e-6))
        if max_range is not None:
            xy = xy[np.linalg.norm(xy, axis=1) <= max_range]
        scan.xy = xy
        yield scan


def build_map(data, cfg, yaw, shift, cell, noise, seed):
    rng = np.random.default_rng(seed + 1000)
    grid = Grid(cell)
    start = transform(np.zeros(3), yaw, shift)
    stream = degraded(scans(data / "mapping", cfg), noise=noise, rng=rng)
    for _ in slam.track(stream, start, grid=grid):
        pass
    with TemporaryDirectory() as work:
        grid.save(Path(work))
        return read_map(Path(work) / "map.yaml")[2]


def run_dataset(args):
    data, layout, seed, methods = args
    data = Path(data)
    cfg = load_config(PROFILE)
    initial = json.loads((data / "initial_pose.json").read_text(encoding="utf-8"))
    truth = read_csv(data / "inspection_truth.csv")
    maps, rows = {}, []
    for name, (deg, shift, cell, max_range, noise) in SCENARIOS.items():
        yaw = np.radians(deg)
        key = (deg, shift, cell, noise)
        if key not in maps:
            maps[key] = build_map(data, cfg, yaw, shift, cell, noise, seed)
        moved = [Pose(p.t, *transform((p.x, p.y, p.yaw), yaw, shift)) for p in truth]
        for method in methods:
            prepare, surface = METHODS[method]
            points = maps[key] if prepare is None else prepare(maps[key])
            rng = np.random.default_rng(seed + 2000)
            stream = degraded(scans(data / "inspection", cfg), max_range, noise, rng)
            out = list(slam.track(stream, transform(initial, yaw, shift), map_points=points,
                                  surface=surface))
            metric = evaluate([Pose(r.t, *r.pose) for r in out], moved)
            rows.append(dict(layout=layout, seed=seed, scenario=name, method=method,
                             ate_cm=100 * metric["ate_xy_rmse_m"],
                             matched=sum(r.status == "matched" for r in out) / len(out)))
    return rows


def summarize(rows, methods):
    keyed = {}
    for r in rows:
        keyed.setdefault((r["layout"], r["scenario"]), {}).setdefault(r["method"], {})[r["seed"]] = r
    head = " | ".join(f"{m} ATE, см" for m in methods)
    lines = [f"| Раскладка | Сценарий | seed | {head} | " +
             " | ".join(f"{m} лучше base" for m in methods[1:]) + " | matched " +
             " → ".join(methods) + " |",
             "|---|---|---:|" + "---:|" * (2 * len(methods) - 1) + "---:|"]
    for (layout, name), by in keyed.items():
        seeds = sorted(by[methods[0]])
        ate = {m: np.array([by[m][s]["ate_cm"] for s in seeds]) for m in methods}
        cells = [f"{ate[m].mean():.2f} ± {ate[m].std(ddof=1):.2f}" for m in methods]
        wins = [f"{int((ate[m] < ate[methods[0]]).sum())}/{len(seeds)}" for m in methods[1:]]
        matched = " → ".join(f"{np.mean([by[m][s]['matched'] for s in seeds]):.3f}" for m in methods)
        lines.append(f"| {layout} | {name} | {len(seeds)} | " + " | ".join(cells + wins) + f" | {matched} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--layout", choices=SEEDS, action="append")
    parser.add_argument("--seeds", type=int, default=10, help="Seeds per layout")
    parser.add_argument("--method", choices=METHODS, action="append")
    parser.add_argument("--only", action="append", choices=SCENARIOS, help="Run these scenarios only")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--reuse", type=Path, help="Take rows of other methods from this results.json")
    args = parser.parse_args()
    methods = args.method or list(METHODS)
    if args.only:
        for name in list(SCENARIOS):
            if name not in args.only:
                del SCENARIOS[name]
    data_root = args.data or args.out / "data"
    jobs = []
    for layout in args.layout or list(SEEDS):
        for seed in SEEDS[layout][:args.seeds]:
            data = data_root / f"{layout}-{seed}"
            if not data.exists():
                generate(data, layout, seed)
            jobs.append((str(data), layout, seed, methods))
    rows, shown = [], list(methods)
    if args.reuse:
        old = [r for r in json.loads(args.reuse.read_text(encoding="utf-8"))
               if r["method"] not in methods and r["scenario"] in SCENARIOS
               and r["seed"] in SEEDS[r["layout"]][:args.seeds] and r["layout"] in (args.layout or SEEDS)]
        rows.extend(old)
        shown = sorted({r["method"] for r in old}, key=list(METHODS).index) + shown
    with ProcessPoolExecutor(args.jobs, initializer=_restrict, initargs=(args.only,)) as pool:
        for part in pool.map(run_dataset, jobs):
            rows.extend(part)
            print(f"{part[0]['layout']} {part[0]['seed']} done", file=sys.stderr)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    text = summarize(rows, shown)
    (args.out / "results.md").write_text(text, encoding="utf-8")
    print(text)


def _restrict(only):
    if only:
        for name in list(SCENARIOS):
            if name not in only:
                del SCENARIOS[name]


if __name__ == "__main__":
    main()
