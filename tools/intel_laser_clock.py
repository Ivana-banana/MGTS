"""Audit Intel FLASER clocks and replay the fixed inspection with sorted IPC stamps."""
import argparse
from collections import defaultdict
from decimal import Decimal
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from mgts_nav.bag import load_config, scans
from mgts_nav.delayed_start import estimate_start
from mgts_nav.grid import read_map
from tools.intel_lab import laser_rows, write_window


def ns(token):
    return int(Decimal(token) * 10**9)


def gaps(stamps):
    intervals = [b - a for a, b in zip(stamps, stamps[1:])]
    return dict(scans=len(stamps), start_ns=stamps[0], end_ns=stamps[-1],
                max_gap_s=max(intervals)/1e9, gaps_over_350ms=sum(x > 350_000_000 for x in intervals),
                covered_fraction=sum(x for x in intervals if x <= 350_000_000) / sum(intervals))


def analyze(rows):
    by_stamp = defaultdict(list)
    control = []
    stamps = [ns(row[-3]) for row in rows]
    logger = [ns(row[-1]) for row in rows]
    high = stamps[0]
    lag = 0
    for t, row in zip(stamps, rows):
        by_stamp[t].append(row)
        if not control or t > high:
            control.append((t, row))
        high = max(t, high)
        lag = max(lag, high - t)
    sorted_rows = [(t, by_stamp[t][0]) for t in sorted(by_stamp)]
    duplicates = [group for group in by_stamp.values() if len(group) > 1]
    conflicting = sum(len({tuple(map(Decimal, row[2:182])) for row in group}) > 1 for group in duplicates)
    offsets = [t - l for t, l in zip(stamps, logger)]
    report = dict(rows=len(rows), adjacent_inversions=sum(b < a for a, b in zip(stamps, stamps[1:])),
                  logger_inversions=sum(b < a for a, b in zip(logger, logger[1:])),
                  logger_duplicates=len(logger)-len(set(logger)),
                  duplicate_stamps=len(duplicates), conflicting_groups=conflicting,
                  max_reorder_lag_s=lag/1e9, offset_ns=[min(offsets), max(offsets)],
                  control=gaps([t for t, _ in control]), sorted=gaps([t for t, _ in sorted_rows]))
    return report, control, sorted_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Verified intel.log.gz")
    parser.add_argument("profile", type=Path)
    parser.add_argument("map", type=Path)
    args = parser.parse_args()
    raw = list(laser_rows(args.source))
    pixels, meta, _ = read_map(args.map)
    config = load_config(args.profile)
    for name, rows in (("all", raw), ("mapping", raw[169:1112]), ("inspection", raw[3420:3498])):
        report, control, sorted_rows = analyze(rows)
        if name == "inspection":
            report["delayed"] = {}
            with TemporaryDirectory() as tmp:
                for mode, selected in (("control", control), ("sorted", sorted_rows)):
                    bag = Path(tmp)/mode
                    write_window(bag, selected)
                    stats = {}
                    try:
                        pose = estimate_start(scans(bag, config), pixels, meta, stats)
                    except ValueError as exc:
                        result = dict(reason=str(exc))
                    else:
                        result = dict(pose=pose.tolist())
                    report["delayed"][mode] = dict(**result, anchors=len(stats["anchors"]),
                                                    prefix=gaps([t for t, _ in selected[:51]]))
        print(json.dumps(dict(window=name, **report), allow_nan=False))


if __name__ == "__main__":
    main()
