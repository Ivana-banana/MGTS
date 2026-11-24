"""Plot scan support and gaps from the inspection CSV, without a reference track."""
import argparse
import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from mgts_nav.quality import inspection_quality


@dataclass
class Sample:
    t: float
    status: str
    inliers: float


def read_samples(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'t', 'status', 'inlier_fraction'}.issubset(reader.fieldnames or []):
            raise ValueError('Expected t, status and inlier_fraction in trajectory CSV')
        samples = []
        for row in reader:
            try:
                sample = Sample(float(row['t']), row['status'], float(row['inlier_fraction']))
                if not isfinite(sample.inliers) or not 0 <= sample.inliers <= 1:
                    raise ValueError('inlier_fraction outside [0, 1]')
            except (TypeError, ValueError) as error:
                raise ValueError(f'CSV line {reader.line_num}: {error}') from error
            samples.append(sample)
        return samples


def plot(samples, gap_threshold_s=None):
    quality = inspection_quality(samples, gap_threshold_s)
    # Without a scan period a gap cannot be told apart from a normal interval.
    period_known = quality['gap_threshold_s'] is not None
    start = samples[0].t
    gaps = {gap['end_s']: gap['start_s'] for gap in quality['scan_gaps']}
    fig = Figure(figsize=(10.5, 4.8), layout='constrained')
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    times, support, predictions = [], [], []
    for sample in samples:
        dt = sample.t - start
        if sample.t in gaps:
            ax.axvspan(gaps[sample.t] - start, dt, facecolor='#ededed',
                       edgecolor='#999999', hatch='///', linewidth=0, zorder=0)
        if times and (sample.t in gaps or sample.status != 'matched' or not period_known):
            times.append(float('nan'))
            support.append(float('nan'))
        if sample.status == 'matched':
            times.append(dt)
            support.append(sample.inliers)
        else:
            predictions.append((dt, sample.inliers))
    ax.plot(times, support, color='#176682', marker='.', markersize=4, linewidth=1.6)
    if predictions:
        x, y = zip(*predictions)
        ax.scatter(x, y, marker='x', color='#a63a16', s=32, clip_on=False, zorder=4)
    duration = samples[-1].t - start
    padding = max(.01, .012 * duration) if duration else .5
    ax.set_xlim(-padding, duration + padding)
    ax.set_ylim(0, 1)
    ax.set_yticks([0., .2, .4, .6, .8, 1.])
    ax.set_ylabel('Доля совпавших точек')
    xlabel = f'Секунды от начала инспекции  ·  t₀ в CSV = {start:.9f} с'
    if not period_known:
        xlabel += '\nПериод сканов неизвестен: разрывы не определяются, точки не соединены'
    ax.set_xlabel(xlabel)
    ax.grid(axis='y', alpha=.25)
    ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Качество привязки сканов к карте', weight='bold', fontsize=16)
    ax.legend(handles=[
        Line2D([], [], color='#176682', marker='.', linestyle='-' if period_known else 'none',
               label='Привязка к карте' if period_known else 'Привязка к карте (период неизвестен)'),
        Line2D([], [], color='#a63a16', marker='x', linestyle='none', label='Только прогноз'),
        Patch(facecolor='#ededed', edgecolor='#999999', hatch='///', label='Пропуск сканов'),
    ], loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False)
    return fig, quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory', type=Path)
    parser.add_argument('png', type=Path)
    parser.add_argument('--gap-s', type=float, default=None,
                        help='scan gap threshold in seconds; default is 3x the median interval')
    args = parser.parse_args()
    try:
        samples = read_samples(args.trajectory)
        fig, quality = plot(samples, args.gap_s)
        args.png.parent.mkdir(parents=True, exist_ok=True)
        with args.png.open('xb') as output:
            fig.savefig(output, format='png', dpi=160)
        # CSV row numbers include the header row. Times are on the input clock.
        lines = {sample.t: i for i, sample in enumerate(samples, 2)}
        for gap in quality['scan_gaps']:
            a, b = gap['start_s'], gap['end_s']
            print(f'gap: {a:.9f}..{b:.9f} s, CSV rows {lines[a]}..{lines[b]}')
        for run in quality['prediction_runs']:
            a, b = run['first_prediction_s'], run['last_prediction_s']
            print(f'prediction_only: {a:.9f}..{b:.9f} s, CSV rows {lines[a]}..{lines[b]}')
    except (OSError, ValueError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
