"""Plot scan-to-map support over inspection time; no reference poses are read.

Matched scans are filled dots joined only to their matched neighbours. Scans kept on
prediction are hollow triangles inside an orange band, missing scans are a hatched grey
band. Every band has a label (P1, G1, ...) that is listed with its CSV line numbers.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw, ImageFont

from .quality import inspection_quality
from .slam import Estimate

SIZE = (1400, 560)
LEFT, RIGHT, TOP, BOTTOM = 90, 30, 60, 110
BLUE, ORANGE, GREY, INK = (31, 99, 181), (222, 120, 20), (150, 150, 150), (30, 30, 30)
ORANGE_BAND, GREY_BAND, GRID = (253, 232, 205), (228, 228, 228), (225, 225, 225)


def read_trajectory(path):
    """Rows of a localize trajectory.csv with their 1-based line numbers in the file."""
    rows = []
    # utf-8-sig: a CSV re-saved by Excel or PowerShell 5.1 starts with a BOM.
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        header = f.readline()
        f.seek(0)
        delimiter = ';' if header.count(';') > header.count(',') else ','
        reader = csv.DictReader(f, delimiter=delimiter)
        missing = {'t', 'status', 'inlier_fraction'} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'{path} lacks columns {sorted(missing)} (header: {header.strip()!r}); '
                             'use trajectory.csv written by localize or solve')
        for row in reader:
            rows.append(dict(line=reader.line_num, t=float(row['t']), status=row['status'],
                             inlier_fraction=float(row['inlier_fraction'])))
    if not rows:
        raise ValueError(f'No scans in {path}')
    return rows


def intervals(rows, quality, t0=None):
    """Prediction runs and scan gaps with times relative to t0 (first scan) and CSV lines."""
    t0 = rows[0]['t'] if t0 is None else t0
    line_at = {row['t']: row['line'] for row in rows}
    found = []
    for run in quality['prediction_runs']:
        found.append(dict(kind='prediction_run', start_s=run['first_prediction_s'],
                          end_s=run['last_prediction_s'], scans=run['scans'],
                          first_line=line_at[run['first_prediction_s']],
                          last_line=line_at[run['last_prediction_s']]))
    for gap in quality['scan_gaps']:
        # The bounding scans exist; nothing was received between their lines.
        found.append(dict(kind='scan_gap', start_s=gap['start_s'], end_s=gap['end_s'], scans=0,
                          first_line=line_at[gap['start_s']], last_line=line_at[gap['end_s']]))
    found.sort(key=lambda item: (item['start_s'], item['kind']))
    counters = {'prediction_run': 0, 'scan_gap': 0}
    for item in found:
        counters[item['kind']] += 1
        item['label'] = ('P' if item['kind'] == 'prediction_run' else 'G') + str(counters[item['kind']])
        item['start_rel_s'] = item['start_s'] - t0
        item['end_rel_s'] = item['end_s'] - t0
    return found


def _font(size):
    for name in ('DejaVuSans.ttf', 'arial.ttf', 'LiberationSans-Regular.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _ticks(lo, hi, count=8):
    span = hi - lo
    raw = span / count
    step = min((m * 10 ** e for e in range(-3, 7) for m in (1, 2, 5) if m * 10 ** e >= raw))
    first = -(-lo // step) * step
    return [first + i * step for i in range(int((hi - first) / step + 1e-9) + 1)]


def render(rows, quality, found, output, title='', t0=None):
    """Save the timeline PNG. Times on the axis are seconds from t0 (first scan)."""
    t0 = rows[0]['t'] if t0 is None else t0
    rel = [row['t'] - t0 for row in rows]
    lo, hi = rel[0], rel[-1]
    if hi - lo < 1.0:  # a single scan or a very short window still needs a readable axis
        mid = (lo + hi) / 2
        lo, hi = mid - 0.5, mid + 0.5
    width, height = SIZE
    x0, x1, y0, y1 = LEFT, width - RIGHT, height - BOTTOM, TOP

    def px(s):
        return x0 + (s - lo) / (hi - lo) * (x1 - x0)

    def py(v):
        return y0 - min(max(v, 0.0), 1.0) * (y0 - y1)

    image = Image.new('RGB', SIZE, 'white')
    draw = ImageDraw.Draw(image)
    small, normal = _font(15), _font(18)

    def span(item):  # intervals clipped by a window still start and end inside the frame
        return px(max(item['start_rel_s'], lo)), px(min(item['end_rel_s'], hi))

    for item in found:  # bands first so points stay on top
        a, b = span(item)
        if item['kind'] == 'prediction_run':
            half = max(3.0, (x1 - x0) / max(len(rows), 1) / 2)
            draw.rectangle([max(a - half, x0), y1, min(b + half, x1), y0], fill=ORANGE_BAND)
        else:
            w, h = max(int(b - a), 1), y0 - y1
            band = Image.new('RGB', (w, h), GREY_BAND)
            hatch = ImageDraw.Draw(band)
            for x in range(-h, w, 12):
                hatch.line([(x, h), (x + h, 0)], fill=GREY)
            image.paste(band, (int(a), y1))
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        draw.line([(x0, py(v)), (x1, py(v))], fill=GRID)
        draw.text((x0 - 10, py(v)), f'{v:.2f}', fill=INK, font=small, anchor='rm')
    for s in _ticks(lo, hi):
        draw.line([(px(s), y0), (px(s), y0 + 6)], fill=INK)
        draw.text((px(s), y0 + 10), f'{s:g}', fill=INK, font=small, anchor='ma')
    draw.rectangle([x0, y1, x1, y0], outline=INK)
    draw.text(((x0 + x1) / 2, y0 + 34), 'seconds from first inspection scan', fill=INK, font=normal, anchor='ma')
    label = Image.new('RGB', (y0 - y1, 24), 'white')
    ImageDraw.Draw(label).text(((y0 - y1) / 2, 12), 'inlier fraction of scan', fill=INK, font=normal, anchor='mm')
    image.paste(label.rotate(90, expand=True), (14, y1))
    # Up to two label rows above the frame; a label that would overlap both is left to the CSV.
    row_end = [-1e9, -1e9]
    for item in found:
        a, b = span(item)
        text = item['label'] + ('*' if item.get('clipped') else '')
        w = draw.textlength(text, font=small)
        x = min(max((a + b) / 2, x0 + w / 2), x1 - w / 2)
        for level in (0, 1):
            if x - w / 2 > row_end[level] + 4:
                row_end[level] = x + w / 2
                draw.text((x, y1 - 4 - 18 * level), text, fill=ORANGE if text[0] == 'P' else GREY,
                          font=small, anchor='md')
                break

    threshold = quality['gap_threshold_s']
    for prev, cur, a, b in zip(rows, rows[1:], rel, rel[1:]):
        joined = prev['status'] == cur['status'] == 'matched'
        # Without a cadence (fewer than three scans) no interval is known to be continuous.
        if joined and threshold is not None and b - a <= threshold:
            draw.line([(px(a), py(prev['inlier_fraction'])), (px(b), py(cur['inlier_fraction']))],
                      fill=BLUE, width=2)
    for row, s in zip(rows, rel):
        x, y = px(s), py(row['inlier_fraction'])
        if row['status'] == 'matched':
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=BLUE)
        else:
            draw.polygon([(x, y - 10), (x - 5, y - 1), (x + 5, y - 1)], outline=ORANGE)  # stands on its value

    matched = quality['matched_scan_fraction']
    draw.text((x0, 20), f'{title}  scans {len(rows)} · matched {matched:.0%}', fill=INK, font=normal, anchor='ls')
    legend_y = height - 28
    draw.ellipse([x0, legend_y - 3, x0 + 6, legend_y + 3], fill=BLUE)
    draw.text((x0 + 14, legend_y), 'matched scan', fill=INK, font=small, anchor='lm')
    draw.polygon([(x0 + 160, legend_y - 5), (x0 + 155, legend_y + 4), (x0 + 165, legend_y + 4)], outline=ORANGE)
    draw.text((x0 + 174, legend_y), 'scan kept on prediction (P)', fill=INK, font=small, anchor='lm')
    draw.rectangle([x0 + 400, legend_y - 7, x0 + 420, legend_y + 7], fill=GREY_BAND, outline=GREY)
    threshold_text = 'n/a' if threshold is None else f'> {threshold:.3g} s'
    draw.text((x0 + 428, legend_y), f'no scans received (G), interval {threshold_text}',
              fill=INK, font=small, anchor='lm')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def write_intervals(found, path):
    fields = ['label', 'kind', 'start_rel_s', 'end_rel_s', 'start_s', 'end_s', 'scans', 'first_line', 'last_line',
              'clipped']
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(found)


def table_path(output):
    output = Path(output)
    return output.with_name(output.stem + '.intervals.csv')


def plot(trajectory, output, window=None, title=None, gap_threshold_s=None):
    """Write PNG and the interval table beside it; return the intervals.

    Intervals, labels and the gap threshold always come from the whole record, so a
    window shows the same P/G labels as the full plot; a clipped interval is marked.
    """
    trajectory, output = Path(trajectory), Path(output)
    table = table_path(output)
    for target in (output, table):
        if target.resolve() == trajectory.resolve():
            raise ValueError(f'Output {target} would overwrite the input trajectory')
    rows = read_trajectory(trajectory)
    t0 = rows[0]['t']
    quality = inspection_quality([Estimate(r['t'], None, r['status'], r['inlier_fraction']) for r in rows],
                                 gap_threshold_s=gap_threshold_s)
    found = intervals(rows, quality, t0)
    label = title if title is not None else str(trajectory.parent.name)
    if window is not None:
        start, end = window
        shown = [row for row in rows if start <= row['t'] - t0 <= end]
        if not shown:
            raise ValueError(f'No scans between {start} and {end} s from the first scan')
        lo, hi = shown[0]['t'] - t0, shown[-1]['t'] - t0
        found = [item for item in found if item['end_rel_s'] >= lo and item['start_rel_s'] <= hi]
        for item in found:
            item['clipped'] = item['start_rel_s'] < lo or item['end_rel_s'] > hi
        rows = shown
        label += f'  lines {rows[0]["line"]}-{rows[-1]["line"]}'
        quality = dict(quality, matched_scan_fraction=sum(r['status'] == 'matched' for r in rows) / len(rows))
    render(rows, quality, found, output, label, t0)
    write_intervals(found, table)
    return found


def demo(output):
    """Synthetic inspection as recorded, with 2 s of lidar messages missing and with 2 s of
    empty lidar messages (the scans stay on prediction); each gets a timeline."""
    from .demo import generate
    from .bag import load_config
    from .stress import fault_bag

    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = Path(__file__).resolve().parents[1] / 'configs' / 'example.json'
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / 'run'
        data = stage / 'data'
        stage.mkdir()
        generate(data)
        cfg = load_config(profile)
        fault_bag(data / 'inspection', data / 'inspection_gap', 'lidar_no_messages', cfg)
        fault_bag(data / 'inspection', data / 'inspection_empty', 'lidar_no_returns', cfg)

        def nav(*args):
            subprocess.run([sys.executable, '-m', 'mgts_nav', *map(str, args)], check=True)
        nav('map', data / 'mapping', '--config', profile, '--out', stage / 'map')
        results = {}
        for name in ('inspection', 'inspection_gap', 'inspection_empty'):
            bag = data / name
            nav('localize', bag, '--config', profile, '--initial-file', data / 'initial_pose.json',
                '--map', stage / 'map' / 'map.yaml', '--out', stage / name)
            found = plot(stage / name / 'trajectory.csv', stage / name / 'timeline.png', title=name)
            results[name] = [item['label'] for item in found]
        stage.rename(output)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('trajectory', type=Path, nargs='?',
                        help='trajectory.csv of localize/solve, or its directory')
    parser.add_argument('--out', type=Path, help='PNG path; the interval table is written beside it as .csv')
    parser.add_argument('--from', dest='start', type=float, help='window start, s from the first scan')
    parser.add_argument('--to', dest='end', type=float, help='window end, s from the first scan')
    parser.add_argument('--gap-s', type=float,
                        help='gap threshold, s (default: three median scan intervals, as in summary.json)')
    parser.add_argument('--demo', type=Path, metavar='OUT',
                        help='build the synthetic demo and a copy with a lidar outage, plot both')
    args = parser.parse_args()
    try:
        if args.demo is not None:
            print(json.dumps(demo(args.demo), indent=2))
            return
        if args.trajectory is None:
            parser.error('give a trajectory.csv or --demo OUT')
        path = args.trajectory / 'trajectory.csv' if args.trajectory.is_dir() else args.trajectory
        window = None
        if args.start is not None or args.end is not None:
            window = (args.start if args.start is not None else float('-inf'),
                      args.end if args.end is not None else float('inf'))
        name = 'timeline.png' if window is None else ('timeline-' + ('start' if args.start is None else f'{args.start:g}') + '-'
                                                 + ('end' if args.end is None else f'{args.end:g}') + '.png')
        out = args.out or path.with_name(name)
        found = plot(path, out, window, gap_threshold_s=args.gap_s)
        print(f'{out}\n{table_path(out)}')
        for item in found:
            print(f"{item['label']}{' (clipped)' if item.get('clipped') else ''}: "
                  f"{item['start_rel_s']:.2f}-{item['end_rel_s']:.2f} s, "
                  f"lines {item['first_line']}-{item['last_line']} of {path}")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
