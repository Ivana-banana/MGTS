"""Measure inspection localization on synthetic layouts, seeds and sensor faults.

For every layout and seed the script generates a fresh mapping/inspection pair,
builds the map with the current tracker, then localizes the clean inspection
and the fault copies from ``mgts_nav.stress``. The analytic truth is read only
for scoring. Two further cases test robustness of the tracker itself: lidar
returns limited to 3 m (as in docs/ambiguity.md) and a start pose shifted by
(0.15 m, -0.10 m, 5 degrees). Results go to ``results.json`` and ``results.md`` in the output
directory; ``--compare`` prints a before/after table for two result files.
"""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mgts_nav.bag import load_config, scans  # noqa: E402
from mgts_nav.demo import PROFILE, generate  # noqa: E402
from mgts_nav.grid import Grid, read_map  # noqa: E402
from mgts_nav.initialization import estimate_initial  # noqa: E402
from mgts_nav.metrics import Pose, evaluate, read_csv, wrap  # noqa: E402
from mgts_nav import slam  # noqa: E402
from mgts_nav.slam import track  # noqa: E402
from mgts_nav.stress import CASES, fault_bag  # noqa: E402

SEEDS = {"office": (17, 101, 202, 303, 404), "corridor": (41, 73, 505, 606, 707)}
START_TOLERANCE = (0.25, np.radians(10))
EXTRA_CASES = ("range_3m", "start_offset")
START_OFFSET = np.array([0.15, -0.10, np.radians(5)])


def limited(stream, max_range):
    for scan in stream:
        scan.xy = scan.xy[np.linalg.norm(scan.xy, axis=1) <= max_range]
        yield scan


def localize(source, cfg, initial, points, truth, max_range=None):
    began = perf_counter()
    stream = scans(source, cfg)
    if max_range is not None:
        stream = limited(stream, max_range)
    rows = list(track(stream, initial, map_points=points))
    elapsed = perf_counter() - began
    metrics = evaluate([Pose(r.t, *r.pose) for r in rows], truth)
    times = np.array([p.t for p in truth])
    reference = np.array([[p.x, p.y] for p in truth])
    errors = [float(np.linalg.norm(r.pose[:2] - reference[np.argmin(abs(times - r.t))])) for r in rows]
    return dict(ate_cm=100 * metrics["ate_xy_rmse_m"],
                rpe_cm=100 * metrics["rpe_translation_rmse_m"],
                rpe_mrad=1000 * metrics["rpe_rotation_rmse_rad"],
                max_err_cm=100 * max(errors), coverage=metrics["coverage"],
                matched=sum(r.status == "matched" for r in rows) / len(rows),
                poses=len(rows), seconds=elapsed)


def auto_start(source, cfg, pixels, meta, truth):
    first = next(scans(source, cfg))
    try:
        pose = estimate_initial(first.xy, pixels, meta)
    except ValueError:
        return "refused", None
    error = float(np.hypot(pose[0] - truth[0].x, pose[1] - truth[0].y))
    correct = error <= START_TOLERANCE[0] and abs(wrap(pose[2] - truth[0].yaw)) <= START_TOLERANCE[1]
    return ("correct" if correct else "false"), 100 * error


def scenario(data, layout, seed, work):
    cfg = load_config(PROFILE)
    began = perf_counter()
    grid = Grid(0.08)
    mapping = list(track(scans(data / "mapping", cfg), np.zeros(3), grid=grid))
    mapping_seconds = perf_counter() - began
    grid.save(work)
    pixels, meta, points = read_map(work / "map.yaml")
    truth = read_csv(data / "inspection_truth.csv")
    map_truth = read_csv(data / "mapping_truth.csv")
    map_ate = evaluate([Pose(r.t, *r.pose) for r in mapping], map_truth)["ate_xy_rmse_m"]
    initial = json.loads((data / "initial_pose.json").read_text(encoding="utf-8"))
    start, start_err = auto_start(data / "inspection", cfg, pixels, meta, truth)
    rows = []
    for case in (*CASES, *EXTRA_CASES):
        source = data / "inspection"
        if case in CASES and case != "clean":
            source = work / case
            fault_bag(data / "inspection", source, case, cfg)
        seed_pose = np.add(initial, START_OFFSET) if case == "start_offset" else initial
        result = localize(source, cfg, seed_pose, points, truth, 3.0 if case == "range_3m" else None)
        rows.append(dict(layout=layout, seed=seed, case=case, map_ate_cm=100 * map_ate,
                         mapping_seconds=mapping_seconds, auto_start=start,
                         auto_start_err_cm=start_err, **result))
    return rows


def run(output, data_root, layouts):
    rows = []
    for layout in layouts:
        for seed in SEEDS[layout]:
            data = data_root / f"{layout}-{seed}"
            if not data.exists():
                generate(data, layout, seed)
            with TemporaryDirectory() as work:
                rows.extend(scenario(data, layout, seed, Path(work)))
            clean = rows[-len(CASES) - len(EXTRA_CASES)]
            print(f"{layout} seed {seed}: clean ATE {clean['ate_cm']:.2f} cm, "
                  f"start {clean['auto_start']}", file=sys.stderr)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (output / "results.md").write_text(table(rows), encoding="utf-8")
    return rows


def groups(rows):
    keyed = {}
    for row in rows:
        keyed.setdefault((row["layout"], row["case"]), []).append(row)
    return keyed


def table(rows):
    lines = ["| Раскладка | Сценарий | ATE, см (ср. ± ст. откл.) | ATE макс. по seed, см | RPE, см | "
             "RPE, мрад | Макс. ошибка, см | matched | Время локализации, с |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for (layout, case), group in groups(rows).items():
        ate = np.array([r["ate_cm"] for r in group])
        mean = lambda key: np.mean([r[key] for r in group])  # noqa: E731
        lines.append(f"| {layout} | {case} | {ate.mean():.2f} ± {ate.std(ddof=1):.2f} | {ate.max():.2f} | "
                     f"{mean('rpe_cm'):.3f} | {mean('rpe_mrad'):.2f} | {mean('max_err_cm'):.2f} | "
                     f"{mean('matched'):.3f} | {mean('seconds'):.3f} |")
    starts = [r for r in rows if r["case"] == "clean"]
    counts = {k: sum(r["auto_start"] == k for r in starts) for k in ("correct", "refused", "false")}
    lines.append("")
    lines.append(f"Автостарт по первому скану: верно {counts['correct']}, отказ {counts['refused']}, "
                 f"ложный {counts['false']} из {len(starts)}. Карта: ATE обхода "
                 f"{np.mean([r['map_ate_cm'] for r in starts]):.2f} см в среднем.")
    return "\n".join(lines) + "\n"


def compare(before, after):
    old, new = groups(before), groups(after)
    lines = ["| Раскладка | Сценарий | ATE до, см | ATE после, см | Δ, см | худший Δ по seed, см | "
             "RPE до → после, см | matched до → после | время до → после, с |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, a in old.items():
        b = new[key]
        delta = [y["ate_cm"] - x["ate_cm"] for x, y in zip(a, b, strict=True)]
        m = lambda rows, k: np.mean([r[k] for r in rows])  # noqa: E731
        lines.append(f"| {key[0]} | {key[1]} | {m(a, 'ate_cm'):.2f} | {m(b, 'ate_cm'):.2f} | "
                     f"{np.mean(delta):+.2f} | {max(delta):+.2f} | {m(a, 'rpe_cm'):.3f} → {m(b, 'rpe_cm'):.3f} | "
                     f"{m(a, 'matched'):.3f} → {m(b, 'matched'):.3f} | "
                     f"{m(a, 'seconds'):.3f} → {m(b, 'seconds'):.3f} |")
    total = lambda rows: sum(r["seconds"] for r in rows)  # noqa: E731
    start = lambda rows: {k: sum(r["auto_start"] == k for r in rows if r["case"] == "clean")  # noqa: E731
                          for k in ("correct", "refused", "false")}
    lines.append("")
    lines.append(f"Суммарное время локализации: {total(before):.2f} → {total(after):.2f} с "
                 f"({100 * (total(after) / total(before) - 1):+.0f} %). "
                 f"Автостарт до: {start(before)}, после: {start(after)}.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="Directory for results.json and results.md")
    parser.add_argument("--data", type=Path, help="Reuse or create generated pairs here")
    parser.add_argument("--layout", choices=SEEDS, action="append")
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("BEFORE", "AFTER"),
                        help="Print a before/after table for two results.json files")
    parser.add_argument("--cell-centres", action="store_true",
                        help="Match against raw cell centres, as before the wall-surface change")
    args = parser.parse_args()
    if args.compare:
        before, after = (json.loads(p.read_text(encoding="utf-8")) for p in args.compare)
        print(compare(before, after))
        return
    if args.cell_centres:
        slam.surface_points = lambda points: np.asarray(points, dtype=float)
    data = args.data or args.out / "data"
    rows = run(args.out, data, args.layout or list(SEEDS))
    print(table(rows))


if __name__ == "__main__":
    main()
