"""Profile Windows map workers against existing benchmark_length stride-1 bags."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from mgts_nav.benchmark import hardware, peak_rss_bytes


def worker():
    from mgts_nav.__main__ import main
    from mgts_nav.grid import Grid
    from mgts_nav.native_grid import NativeGrid

    stats = {}
    evidence = NativeGrid.evidence
    save = Grid.save

    def measured_export(grid):
        stats["before_export_mib"] = peak_rss_bytes() / 2**20
        start = perf_counter()
        values = evidence(grid)
        stats["export_s"] = perf_counter() - start
        return values

    def measured_save(grid, directory):
        stats["before_image_mib"] = peak_rss_bytes() / 2**20
        stats["cells"] = len(grid.evidence)
        save(grid, directory)

    NativeGrid.evidence = measured_export
    Grid.save = measured_save
    del sys.argv[1]
    main()
    stats["peak_mib"] = peak_rss_bytes() / 2**20
    print(json.dumps(stats))


def benchmark(source, dll, output):
    if sys.platform != "win32":
        raise ValueError("Requires Windows peak working set")
    output.mkdir(parents=True)
    rows = []
    for length in ("short", "long"):
        bag = source / f"{length}-stride1" / "mapping"
        expected = None
        for mode in (("python", "cpp") if length == "short" else ("cpp", "python")):
            dest = output / f"{length}-{mode}"
            command = [sys.executable, "-m", "tools.benchmark_grid_memory", "--worker", "map", str(bag),
                       "--config", "configs/example.json", "--resolution", ".08", "--out", str(dest)]
            if mode == "cpp":
                command.extend(["--grid-dll", str(dll)])
            start = perf_counter()
            log = subprocess.check_output(command, text=True)
            wall = perf_counter() - start
            stats = json.loads(log.splitlines()[-1])
            hashes = {name: hashlib.sha256((dest / name).read_bytes()).hexdigest()
                      for name in ("map.pgm", "map.yaml", "trajectory.csv")}
            if expected is not None and hashes != expected:
                raise ValueError(f"{length}: map products differ")
            expected = hashes
            summary = json.loads((dest / "summary.json").read_bytes())
            rows.append(dict(length=length, mode=mode, scans=summary["scans"], wall_s=wall, sha256=hashes, **stats))
            print(json.dumps(rows[-1]), flush=True)
    report = dict(machine=hardware(), dll_sha256=hashlib.sha256(dll.read_bytes()).hexdigest(), rows=rows)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("source", type=Path, help="Output of tools.benchmark_length")
        parser.add_argument("dll", type=Path)
        parser.add_argument("output", type=Path)
        args = parser.parse_args()
        benchmark(args.source, args.dll, args.output)
