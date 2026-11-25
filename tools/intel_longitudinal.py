"""Profile longitudinal translation from the first Intel inspection scan to three RBPF epochs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from mgts_nav.metrics import read_csv, relative_pose
from mgts_nav.slam import world
from tools.intel_lab import laser_rows, points


def alignment(first, first_tree, current, pose):
    transformed = world(current, pose)
    forward, targets = first_tree.query(transformed)
    backward, sources = cKDTree(transformed).query(first)
    score = (np.minimum(forward, .25).mean() + np.minimum(backward, .25).mean()) / 2
    mutual = (sources[targets] == np.arange(len(current))) & (forward <= .25)
    return float(score), int(mutual.sum())


def profile(source, reference, output):
    refs = read_csv(reference)
    if len(refs) != 4 or any(b.t <= a.t for a, b in zip(refs, refs[1:])):
        raise ValueError("Expected the four increasing inspection reference times from issue 110")
    times = {p.t for p in refs}
    clouds = {}
    for row in laser_rows(source):
        t = float(row[-3])
        if t in times:
            if t in clouds:
                raise ValueError(f"Duplicate reference scan at {t}")
            clouds[t] = points(np.asarray(row[2:182], float))[:, :2].astype(float)
    if clouds.keys() != times:
        raise ValueError("Missing raw FLASER at a reference epoch")
    first = clouds[refs[0].t]
    tree = cKDTree(first)
    xs = np.arange(-100, 401) / 100
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    with (output / "profile.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["elapsed_s", "x_m", "mean_capped_distance_m", "mutual_pairs"])
        for ref in refs[1:]:
            elapsed = ref.t - refs[0].t
            x_ref, y, yaw = relative_pose(refs[0], ref)
            current = clouds[ref.t]
            measurements = [alignment(first, tree, current, [x, y, yaw]) for x in xs]
            for x, (score, count) in zip(xs, measurements):
                writer.writerow([elapsed, x, score, count])
            scores = np.array([s for s, _ in measurements])
            best = int(scores.argmin())
            near = scores <= scores[best] + .01
            lo = hi = best
            while lo > 0 and near[lo - 1]:
                lo -= 1
            while hi + 1 < len(xs) and near[hi + 1]:
                hi += 1
            rbpf = alignment(first, tree, current, [x_ref, y, yaw])
            print(json.dumps(dict(elapsed_s=elapsed, sizes=[len(first), len(current)], x_rbpf=x_ref,
                                  best_x=xs[best], best=measurements[best], zero=measurements[100],
                                  rbpf=rbpf, near=[xs[lo], xs[hi]])))
            line, = ax.plot(xs, scores * 100, label=f"{elapsed:.2f} с")
            ax.plot(x_ref, rbpf[0]*100, "x", color=line.get_color())
            ax.plot(xs[best], scores[best]*100, "o", ms=4, color=line.get_color())
    ax.set(xlabel="Продольный сдвиг в осях первого скана, м", ylabel="Среднее ограниченное расстояние, см")
    ax.grid(alpha=.22)
    ax.legend(title="Интервал")
    fig.tight_layout()
    fig.savefig(output / "profile.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "reference", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    profile(args.source, args.reference, args.output)
