"""Plot the measured XY error on the four 300-scan HEAP-1 survey runs."""
import argparse
import json
from pathlib import Path

from matplotlib.figure import Figure
import numpy as np

from mgts_nav.metrics import read_csv


def plot(reference, experiments, output):
    truth = np.array(read_csv(reference))
    scores = json.loads((experiments/'scores.json').read_bytes())
    fig = Figure(figsize=(10, 6), layout='constrained')
    axes = fig.subplots(2, 2, sharex=True, sharey=True)
    modes = [('raw', 'Штатный обход'), ('prior', 'Прогноз скорости'),
             ('gyro', 'Компенсация вращения'), ('both', 'Прогноз и компенсация')]
    for ax, (mode, title) in zip(axes.flat, modes, strict=True):
        track = np.array(read_csv(experiments/mode/'map/trajectory.csv'))
        np.testing.assert_allclose(track[:, 0], truth[:, 0], rtol=0, atol=1e-6)
        xy = np.column_stack([np.interp(truth[:, 0], track[:, 0], component)
                              for component in track[:, 1:3].T])
        err = np.linalg.norm(xy-truth[:, 1:3], axis=1)
        stats = scores[mode]['mapping']
        np.testing.assert_allclose([err[-1], np.sqrt(np.mean(err**2))],
                                   [stats['final_xy_m'], stats['ate_m']], rtol=1e-8)
        t = truth[:, 0]-truth[0, 0]
        ax.plot(t, err*100, color='#176b70', lw=1.6)
        ax.plot(t[-1], err[-1]*100, 'o', color='#176b70', ms=4)
        ax.set_title(title, loc='left', fontweight='bold', fontsize=13)
        ax.text(.03, .91, f'В конце {err[-1]*100:.1f} см'.replace('.', ','), transform=ax.transAxes)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=.25)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('HEAP-1 · ошибка положения на 300 сканах обхода', fontsize=17)
    fig.supylabel('Ошибка XY относительно CPT7, см')
    fig.supxlabel('Время от начала обхода, с')
    with output.open('xb') as stream:
        fig.savefig(stream, format='png', dpi=160)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'experiments', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    plot(args.reference, args.experiments, args.output)
