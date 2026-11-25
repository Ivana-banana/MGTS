"""Evaluate delayed initialization after a broken scan chain; output only the localized suffix."""
import argparse
import csv
import json
from itertools import chain, islice
from pathlib import Path
from tempfile import TemporaryDirectory

from mgts_nav.bag import load_config, scans
from mgts_nav.delayed_start import MAX_WINDOW_SCANS, estimate_prepared, prepare_window
from mgts_nav.grid import read_map
from mgts_nav.slam import track


def search(stream, pixels, meta, points, method):
    stream = iter(stream)
    window = []
    skipped = []
    attempts = []
    for _ in range(3):
        window.extend(islice(stream, MAX_WINDOW_SCANS - len(window)))
        prepared = prepare_window(window, local_method=method)
        diagnostics = {}
        attempt = dict(start_index=len(skipped), scans=len(window), first_t=window[0].t if window else None)
        attempts.append(attempt)
        try:
            pose = estimate_prepared(prepared, pixels, meta, diagnostics)
        except ValueError as exc:
            attempt['reason'] = str(exc)
            pair = prepared.rejected_pair
            if pair is None:
                break
            attempt['rejected_pair'] = pair
            if len(attempts) == 3:
                break
            advance = pair['indices'][1]
            if len(window[advance].xy) < 12:
                advance += 1
            skipped.extend(s.t for s in window[:advance])
            window = window[advance:]
        else:
            report = dict(outcome='accepted', attempts=attempts, uncovered_t=skipped,
                          start_t=window[0].t, start=pose.tolist())
            return report, track(chain(window, stream), pose, map_points=points)
    skipped.extend(s.t for s in chain(window, stream))
    return dict(outcome='refused', attempts=attempts, uncovered_t=skipped), ()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bag', 'profile', 'map', 'out'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--method', choices=('pairwise', 'keyframes'), default='pairwise')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Choose a new output directory')
    pixels, meta, points = read_map(args.map)
    result, rows = search(scans(args.bag, load_config(args.profile)), pixels, meta, points, args.method)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp)/'result'
        stage.mkdir()
        if result['outcome'] == 'accepted':
            with (stage/'trajectory.csv').open('w', newline='', encoding='utf8') as file:
                writer = csv.writer(file)
                writer.writerow(('t', 'x', 'y', 'yaw', 'status'))
                for row in rows:
                    writer.writerow((row.t, *row.pose, row.status))
        (stage/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
        stage.rename(args.out)
    return 0 if result['outcome'] == 'accepted' else 3


if __name__ == '__main__':
    raise SystemExit(main())
