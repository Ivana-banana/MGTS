"""Time repeated and shared local estimation on the four HEAP-1 inspection windows."""
import argparse
from itertools import islice
from pathlib import Path
from time import perf_counter

from mgts_nav.bag import load_config, scans
from mgts_nav.delayed_start import prepare_window, estimate_prepared
from mgts_nav.grid import read_map


def measure(window, rasters, shared):
    results = []
    elapsed_local = 0.
    prepared = None
    start = perf_counter()
    for pixels, meta in rasters:
        if not shared or prepared is None:
            tick = perf_counter()
            prepared = prepare_window(window)
            elapsed_local += perf_counter() - tick
        stats = {}
        try:
            pose = estimate_prepared(prepared, pixels, meta, stats)
        except ValueError as exc:
            results.append((stats, str(exc)))
        else:
            results.append((stats, pose.tolist()))
    return results, elapsed_local, perf_counter() - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("revisit", "holdout", "sections"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    rasters = [read_map(args.sections / f"section-{i}" / "map.yaml")[:2] for i in range(0, 300, 40)]
    for data, first, count in ((args.revisit, 2560, 1), (args.holdout, 2620, 3)):
        stream = scans(data / "inspection", load_config(data / "profile.json"))
        for i in range(count):
            window = list(islice(stream, 60))
            if len(window) != 60:
                raise ValueError("Expected complete 60-scan windows")
            repeated, local_before, total_before = measure(window, rasters, False)
            shared, local_after, total_after = measure(window, rasters, True)
            if repeated != shared:
                raise ValueError(f"Different diagnostics at frame {first + 60*i}")
            print(first + 60*i, local_before, local_after, total_before, total_after, flush=True)
