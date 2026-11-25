"""Plot the external HEAP-1 moving-window validation in its original map frame."""
import argparse
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from mgts_nav.grid import read_map
from mgts_nav.metrics import read_csv


def plot(solution, reference, output):
    pixels, meta, _ = read_map(solution / 'map/map.yaml')
    mapping = np.array(read_csv(solution / 'map/trajectory.csv'))
    inspection = np.array(read_csv(solution / 'inspection/trajectory.csv'))
    truth = np.array(read_csv(reference))
    summary = json.loads((solution/'inspection/summary.json').read_bytes())
    if (summary['initialization'] != 'delayed' or len(mapping) != 40 or len(inspection) != 60
            or truth.shape != inspection.shape
            or abs(mapping[0, 0] - 1731588427.081529) > 1e-6
            or abs(inspection[0, 0] - 1731588668.081259) > 1e-6
            or not np.allclose(truth[:, 0], inspection[:, 0], rtol=0, atol=1e-6)):
        raise ValueError('Expected paired timestamps from the published HEAP-1 moving window')
    xy = inspection[:, 1:3]
    ref = truth[:, 1:3]
    errors = 100 * np.linalg.norm(xy-ref, axis=1)
    rmse = np.sqrt(np.mean(errors**2))
    figure = Figure(figsize=(10.5, 6.7), layout='constrained')
    FigureCanvasAgg(figure)
    layout = figure.add_gridspec(2, 2, height_ratios=(2.25, 1))
    overview = figure.add_subplot(layout[0, 0])
    detail = figure.add_subplot(layout[0, 1])
    error = figure.add_subplot(layout[1, :])
    x, y = meta['origin'][:2]
    h, w = pixels.shape
    cell = meta['resolution']
    overview.imshow(pixels, cmap='gray', vmin=0, vmax=255, origin='upper', interpolation='nearest',
                    extent=(x, x+w*cell, y, y+h*cell))
    teal, blue, red = '#007b61', '#114fa3', '#b13f16'
    overview.plot(*mapping[:, 1:3].T, color=teal, lw=2.2, label='Обход')
    overview.plot(*xy.T, color=blue, lw=2.5, label='Инспекция')
    area = np.vstack([mapping[:, 1:3], xy, ref])
    for axis, values in ((overview.set_xlim, area[:, 0]), (overview.set_ylim, area[:, 1])):
        axis(values.min()-9, values.max()+9)
    overview.set_title('Фрагмент карты обхода')
    overview.legend(loc='lower right')

    detail.plot(*xy.T, '.-', color=blue, lw=1.1, ms=3, label='Оценка MGTS')
    detail.plot(*ref.T, '--', color=red, lw=1.6, label='Опора CPT7')
    detail.scatter(*xy[0], s=56, facecolors='white', edgecolors=blue, zorder=3, label='Первая оценка')
    detail.set_title('Инспекция крупно')
    detail.legend(loc='upper left', fontsize=9)
    for axis in (overview, detail):
        axis.set(xlabel='X карты, м', ylabel='Y карты, м', aspect='equal')
    detail.grid(alpha=.22)

    elapsed = inspection[:, 0]-inspection[0, 0]
    error.plot(elapsed, errors, color=blue, lw=1.4)
    error.axhline(rmse, color=red, ls='--', lw=1.1, label=f'СКО {rmse:.2f} см'.replace('.', ','))
    error.set(xlabel='Время от начала инспекции, с', ylabel='Ошибка XY, см',
              xlim=(0, elapsed[-1]), ylim=(0, errors.max()*1.15))
    error.grid(alpha=.22)
    error.legend(loc='upper right')
    figure.suptitle(f'GrandTour HEAP-1 · повторное посещение · {summary["status"]["matched"]}/60 привязок',
                    fontsize=16)
    figure.supxlabel('Robotic Systems Lab, ETH Zürich · Velodyne VLP-16 · 14 ноября 2024 · CC BY-SA 4.0',
                     fontsize=9, color='#484848')
    with output.open('xb') as target:
        figure.savefig(target, format='png', dpi=145, metadata={
            'Source': 'ETH Zurich Robotic Systems Lab, GrandTour HEAP-1, 2024-11-14',
            'License': 'CC BY-SA 4.0', 'Description': 'MGTS map and trajectory compared to CPT7',
        })
    return rmse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('solution', 'reference', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(f'XY RMSE: {plot(args.solution, args.reference, args.output):.4f} cm')
