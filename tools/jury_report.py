"""Build a self-contained offline HTML briefing from a completed solve."""
import argparse
import base64
from html import escape
import json
import math
from pathlib import Path

from mgts_nav.metrics import evaluate, read_csv


def report(solution, output, truth=None, elapsed_file=None):
    summary = json.loads((solution / 'inspection/summary.json').read_bytes())
    x, y, yaw = summary['final_pose']
    matched = summary['status'].get('matched', 0)
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Отчёт инспекции</title>',
             '<style>body{max-width:56rem;margin:2rem auto;padding:0 1rem;font:18px/1.5 system-ui}'
             'img{max-width:100%}code{overflow-wrap:anywhere}small{color:#444}</style>',
             '<h1>Инспекция по карте обхода</h1>', f'<p>Комплект <code>{escape(str(solution))}</code></p>']
    for title, image in [('Карта и обход', 'map/route.png'), ('Маршрут инспекции', 'inspection/route.png')]:
        data = base64.b64encode((solution / image).read_bytes()).decode('ascii')
        parts.append(f'<h2>{title}</h2><img alt="{title}" src="data:image/png;base64,{data}"><br><small>{image}</small>')
    parts.extend(['<h2>Привязка</h2>', f'<p>Старт {escape(summary["initialization"])} '
                  f'{escape(str(summary["initial_pose"]))} (м, м, рад). '
                  f'Конечная поза ({x:.3f} м, {y:.3f} м, {yaw:.3f} рад). '
                  f'Сопоставлено {matched} из {summary["scans"]} сканов.</p>',
                  '<small>inspection/summary.json</small>'])
    quality = summary['quality']
    runs = quality['prediction_runs']
    if runs:
        loss = max(runs, key=lambda row: row['scans'])
        parts.append(f'<p>Самая длинная серия прогноза без привязки — {loss["scans"]} сканов '
                     f'с {loss["first_prediction_s"]:.3f} по {loss["last_prediction_s"]:.3f} с.</p>')
    else:
        parts.append('<p>Все зарегистрированные сканы прошли без серий prediction_only.</p>')
    if quality['scan_gaps']:
        gap = max(quality['scan_gaps'], key=lambda row: row['duration_s'])
        parts.append(f'<p>Самый длинный пропуск сканов — {gap["duration_s"]:.3f} с '
                     f'после {gap["start_s"]:.3f} с.</p>')
    parts.append('<h2>Сравнение с кинематикой</h2>')
    reference = read_csv(truth) if truth is not None else None
    for name, title in [('inspection', 'Локализация'), ('baseline', 'Штатная одометрия')]:
        path = solution / name / 'trajectory.csv'
        if name == 'baseline' and not path.exists():
            parts.append('<p>В комплекте отсутствует baseline/trajectory.csv. Кинематическое сравнение недоступно.</p>')
            continue
        poses = read_csv(path)
        if reference:
            metrics = evaluate(poses, reference)
            parts.append(f'<p>{title}. Покрыто {metrics["matched_samples"]} из {len(reference)} меток.')
            if metrics['matched_samples']:
                parts.append(f' ATE XY {metrics["ate_xy_rmse_m"]:.3f} м. Максимум ошибки '
                             f'{metrics["max_xy_error_m"]:.3f} м на {metrics["max_xy_error_t"]:.3f} с.')
            command = f'python -m mgts_nav.metrics "{path}" "{truth}"'
            parts.append(f'</p><small><code>{escape(command)}</code></small>')
        else:
            parts.append(f'<p>{title}. Последняя позиция по {name}/trajectory.csv '
                         f'({poses[-1].x:.3f}, {poses[-1].y:.3f}) м. Эталон не предоставлен.</p>')
    parts.append('<h2>Время и повтор</h2>')
    if elapsed_file is not None:
        seconds = float(elapsed_file.read_text(encoding='utf-8-sig').strip())
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Elapsed solve time must be finite and positive')
        parts.append(f'<p>Время solve по внешнему таймеру {seconds:.2f} с '
                     f'(<code>{escape(str(elapsed_file))}</code>).</p>')
    else:
        parts.append('<p>Замер времени solve отсутствует.</p>')
    if (solution / 'solve-options.json').is_file():
        options = json.loads((solution / 'solve-options.json').read_bytes())
        dll = ' --grid-dll PATH_TO_ORIGINAL_DLL' if options['grid_backend'] == 'cpp' else ''
        repeat = f'python -m tools.repeat_solve "{solution}" MAPPING_BAG INSPECTION_BAG NEW_OUTPUT{dll}'
        parts.append(f'<p>Подставьте исходные записи и новый выходной каталог.</p><code>{escape(repeat)}</code>')
    else:
        parts.append('<p>Этот комплект предшествует сохранению solve-options.json. Для повтора нужен журнал запуска.</p>')
    parts.append('</html>')
    with output.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(parts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('solution', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--truth', type=Path, help='Independent t,x,y,yaw in the same map frame')
    parser.add_argument('--elapsed-file', type=Path, help='Wall time of the solve in seconds, ASCII decimal')
    args = parser.parse_args()
    try:
        report(args.solution, args.output, args.truth, args.elapsed_file)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'{error}\n')
