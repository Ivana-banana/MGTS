"""Прогнать демо на синтетике: карта, локализация, сравнение со штатной одометрией и эталоном.

Результат: comparison.png (наглядное сравнение), map/ и inspection/route.png, metrics.json.
"""
import argparse
import json
from pathlib import Path
import subprocess
from importlib.resources import as_file
from tempfile import TemporaryDirectory

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .demo import PROFILE, generate
from .solve import module_command
from .grid import read_map
from .metrics import evaluate, read_csv

TRUTH, BASELINE, OURS, MAPPING, START = '#1f2a44', '#8a6100', '#d1492a', '#0074b8', '#18803a'


def centimetres(value):
    return 'нет данных' if value is None else f'{100 * value:.1f} см'.replace('.', ',')


def comparison(map_yaml, output, mapping, truth, baseline, inspection, metrics):
    """Map with both inspection estimates and the truth, plus a close-up of the inspection."""
    pixels, meta, _ = read_map(map_yaml)
    h, w = pixels.shape
    x0, y0 = meta['origin'][:2]
    extent = [x0, x0 + w * meta['resolution'], y0, y0 + h * meta['resolution']]

    def xy(poses):
        return np.array([(p.x, p.y) for p in poses]).T

    ate_ours = centimetres(metrics['inspection']['ate_xy_rmse_m'])
    ate_odom = centimetres(metrics['baseline']['ate_xy_rmse_m'])
    figure = Figure(figsize=(16, 7), dpi=110, layout='constrained')
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    left.imshow(pixels, cmap='gray', vmin=0, vmax=255, extent=extent, interpolation='nearest')
    left.plot(*xy(mapping), color=MAPPING, lw=1.5, alpha=.6, label='Обход для построения карты')
    for axis in (left, right):
        axis.plot(*xy(truth), color=TRUTH, lw=2.5, ls=':', label='Эталон инспекции')
        axis.plot(*xy(baseline), color=BASELINE, lw=2, ls='--', label=f'Штатная одометрия · ATE {ate_odom}')
        axis.plot(*xy(inspection), color=OURS, lw=2, label=f'Локализация по карте · ATE {ate_ours}')
        axis.plot(truth[0].x, truth[0].y, 'o', color=START, ms=9, label='Старт инспекции', zorder=5)
        axis.set_aspect('equal')
        axis.set_xlabel('x, м')
        axis.set_ylabel('y, м')
    left.set_title('Карта по обходу: чёрный — стены, белый — свободно, серый — не видно')
    left.legend(loc='upper left', fontsize=9, framealpha=.9)
    right.set_title('Инспекция крупно: чем ближе к эталону, тем лучше')
    right.set_aspect('equal', adjustable='datalim')  # fill the panel, leaving room for the legend
    right.grid(alpha=.3)
    right.legend(loc='lower center', fontsize=10, framealpha=.9)
    figure.suptitle(f'Ошибка положения на инспекции (ATE, RMSE): локализация {ate_ours}, '
                    f'штатная одометрия {ate_odom}', fontsize=15)
    figure.savefig(output)


def report(output, metrics):
    ours, odom, mapping = metrics['inspection'], metrics['baseline'], metrics['mapping']
    lines = [
        f'Готово: {output}',
        f"Локализация по карте: ATE {centimetres(ours['ate_xy_rmse_m'])}, "
        f"покрыто {ours['matched_samples']} из {ours['truth_samples']} меток эталона ({ours['coverage']:.0%})",
        f"Штатная одометрия:    ATE {centimetres(odom['ate_xy_rmse_m'])}, "
        f"покрыто {odom['matched_samples']} из {odom['truth_samples']} ({odom['coverage']:.0%})",
        f"Траектория обхода при построении карты: ATE {centimetres(mapping['ate_xy_rmse_m'])}",
    ]
    if ours['ate_xy_rmse_m'] and odom['ate_xy_rmse_m']:
        ratio = odom['ate_xy_rmse_m'] / ours['ate_xy_rmse_m']
        times = f'{max(ratio, 1 / ratio):.1f}'
        if times == '1.0':
            lines.append('Ошибка локализации примерно такая же, как у одометрии.')
        elif ratio > 1:
            lines.append(f"Ошибка локализации в {times.replace('.', ',')} раза меньше, чем у одометрии.")
        else:
            lines.append(f"Ошибка локализации больше, чем у одометрии, в {times.replace('.', ',')} раза.")
    lines += ['Посмотреть:',
              f'  сравнение с эталоном  {output / "comparison.png"}',
              f'  маршрут инспекции     {output / "inspection" / "route.png"}',
              f'  карта и обход         {output / "map" / "route.png"}',
              f'  все метрики (JSON)    {output / "metrics.json"}']
    return '\n'.join(lines)


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp, as_file(PROFILE) as profile:
        stage = Path(tmp) / 'run'
        stage.mkdir()
        data = stage / 'data'
        generate(data)

        def nav(*args):
            # Each step saves its summary.json; its JSON on stdout would only bury the report.
            subprocess.run(module_command('mgts_nav', *args), check=True,
                           stdout=subprocess.DEVNULL)
        nav('map', data / 'mapping', '--config', profile, '--out', stage / 'map')
        options = [data / 'inspection', '--config', profile, '--initial-file', data / 'initial_pose.json']
        nav('baseline', *options, '--out', stage / 'baseline')
        nav('localize', *options, '--map', stage / 'map' / 'map.yaml', '--out', stage / 'inspection')
        truth = read_csv(data / 'inspection_truth.csv')
        tracks = {name: read_csv(stage / name / 'trajectory.csv') for name in ('baseline', 'inspection', 'map')}
        metrics = {name: evaluate(tracks[name], truth) for name in ('baseline', 'inspection')}
        metrics['mapping'] = evaluate(tracks['map'], read_csv(data / 'mapping_truth.csv'))
        (stage / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
        comparison(stage / 'map' / 'map.yaml', stage / 'comparison.png', tracks['map'], truth,
                   tracks['baseline'], tracks['inspection'], metrics)
        stage.rename(output)
    print(report(output, metrics))
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out', type=Path, help='новая папка для результатов, например out/my-first-demo')
    try:
        run(parser.parse_args().out)
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted\n')
    except (OSError, ValueError) as error:
        parser.exit(2, f'{error}\n')
