"""Peak memory and time of loading a large map for localization, per point model.

Writes a synthetic 60 Mpix map (8000 x 7500 cells of 5 cm, about 2.4 million
occupied cells on a lattice of walls), then in a fresh process for each model
reads the map, builds the localization structures and matches one scan.
Two peaks are reported: the process RSS (``resource.getrusage``, Linux), which
is dominated by ``read_map``, and the ``tracemalloc`` peak of everything after
the map points exist (point model, KD-tree, first match), with ``--trace``.

    python tools/benchmark_surface_memory.py out/bigmap
"""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter
import tracemalloc

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_map(directory):
    from PIL import Image
    import yaml
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "map.yaml").exists():
        return
    pixels = np.full((7500, 8000), 254, dtype=np.uint8)
    pixels[::50, :] = 0
    pixels[:, ::50] = 0
    Image.fromarray(pixels).save(directory / "map.pgm")
    meta = dict(image="map.pgm", resolution=0.05, origin=[0.0, 0.0, 0.0], negate=0,
                occupied_thresh=0.65, free_thresh=0.196)
    (directory / "map.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")


def child(directory, model, trace):
    from mgts_nav import slam
    from mgts_nav.bag import Scan
    from mgts_nav.grid import read_map
    began = perf_counter()
    points = read_map(directory / "map.yaml")[2]
    loaded = perf_counter()
    if trace:
        tracemalloc.start()
    if model == "mean3":
        sys.path.insert(0, str(ROOT / "tools"))
        from surface_scenes import mean3
        points = mean3(points)
    # A scan inside one 2.5 m room of the lattice, seen from its centre.
    angles = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    rays = np.column_stack([np.cos(angles), np.sin(angles)])
    distance = 1.25 / np.maximum(np.abs(rays[:, 0]), np.abs(rays[:, 1]))
    scan = Scan(0.0, rays * distance[:, None], np.zeros(2), None, None)
    pose = np.array([101.25 + .02, 101.25 - .01, 0.01])
    estimate = next(slam.track(iter([scan]), pose, map_points=points, surface=model == "surface"))
    done = perf_counter()
    setup_peak = tracemalloc.get_traced_memory()[1] / 2**20 if trace else None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps(dict(model=model, points=len(points), read_s=round(loaded - began, 3),
                          setup_and_match_s=round(done - loaded, 3), setup_peak_mib=setup_peak and round(setup_peak, 1),
                          rss_peak_mib=round(peak, 1), status=estimate.status)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--child", choices=["base", "surface", "mean3"])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--trace", action="store_true",
                        help="Also report the tracemalloc peak (slows the mean3 model about 3x)")
    parser.add_argument("--with-mean3", action="store_true", help="Also measure the first PR version")
    args = parser.parse_args()
    if args.child:
        child(args.out, args.child, args.trace)
        return
    write_map(args.out)
    models = ["base", "surface"] + (["mean3"] if args.with_mean3 else [])
    for _ in range(args.repeat):
        for model in models:
            subprocess.run([sys.executable, __file__, str(args.out), "--child", model]
                           + (["--trace"] if args.trace else []), check=True)


if __name__ == "__main__":
    main()
