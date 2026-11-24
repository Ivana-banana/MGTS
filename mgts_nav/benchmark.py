"""Measure fresh mapping/localization processes at three occupancy resolutions."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter


def peak_rss_bytes():
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t),
                ("peak_paged", ctypes.c_size_t), ("paged", ctypes.c_size_t),
                ("peak_nonpaged", ctypes.c_size_t), ("nonpaged", ctypes.c_size_t),
                ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t),
            ]

        counts = Counters()
        counts.cb = ctypes.sizeof(counts)
        handle = ctypes.windll.kernel32.GetCurrentProcess
        handle.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not query(handle(), ctypes.byref(counts), counts.cb):
            raise ctypes.WinError()
        return counts.peak_working_set
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def hardware():
    cpu = platform.processor() or platform.machine()
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    elif sys.platform.startswith("linux") and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.partition(":")[2].strip()
                break
    return dict(cpu=cpu, logical_cpus=os.cpu_count(), architecture=platform.machine(),
                os=platform.platform(), python=platform.python_version())


def measure(arguments):
    start = perf_counter()
    output = subprocess.check_output(
        [sys.executable, "-m", "mgts_nav.benchmark", "--worker", *map(str, arguments)],
        text=True,
    )
    elapsed = perf_counter() - start
    peak = int(output.splitlines()[-1])
    return dict(wall_s=elapsed, peak_rss_mib=peak / 2**20)


def benchmark(demo, config):
    from PIL import Image
    from .metrics import evaluate, read_csv

    truth = read_csv(demo / "inspection_truth.csv")
    results = []
    with TemporaryDirectory() as work:
        for resolution in (0.05, 0.08, 0.12):
            directory = Path(work) / str(resolution)
            map_dir, loc_dir = directory / "map", directory / "loc"
            mapping = measure(["map", demo / "mapping", "--resolution", resolution,
                               "--config", config, "--out", map_dir])
            localization = measure(["localize", demo / "inspection", "--map", map_dir / "map.yaml",
                                    "--initial-file", demo / "initial_pose.json",
                                    "--config", config, "--out", loc_dir])
            for phase, output in [(mapping, map_dir), (localization, loc_dir)]:
                n = json.loads((output / "summary.json").read_text())["scans"]
                phase["scans_per_s"] = n / phase["wall_s"]
            metrics = evaluate(read_csv(loc_dir / "trajectory.csv"), truth)
            with Image.open(map_dir / "map.pgm") as image:
                shape = list(image.size)
            results.append(dict(resolution_m=resolution, map_size_px=shape,
                                pgm_bytes=(map_dir / "map.pgm").stat().st_size,
                                mapping=mapping, localization=localization,
                                ate_xy_rmse_m=metrics["ate_xy_rmse_m"], coverage=metrics["coverage"]))
    return dict(hardware=hardware(), measurements=results)


def main():
    if sys.argv[1:2] == ["--worker"]:
        from .__main__ import main as navigation_main
        del sys.argv[1]
        navigation_main()
        print(peak_rss_bytes())
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", type=Path)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs" / "example.json")
    args = parser.parse_args()
    print(json.dumps(benchmark(args.demo, args.config), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
