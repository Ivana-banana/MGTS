"""Evaluate mgts_nav.self_check on profiles with a known error and on correct profiles.

    python tools/evaluate_self_check.py out/self-check-eval

Data: tools/foreign_bags.py (scenarios a, c, d, e) and mgts_nav.demo (office and corridor,
five seeds each), generated under OUT/data unless already there. Errors are injected into
the correct profile, so the true correction is known exactly. Writes OUT/results.csv,
OUT/results.json and one report per case in OUT/reports/.
"""
import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mgts_nav.bag import load_config  # noqa: E402
from mgts_nav.demo import generate  # noqa: E402
from mgts_nav.self_check import self_check, with_rotation  # noqa: E402

SEEDS = (17, 23, 31, 41, 73)
TILT_TOLERANCE_DEG = 3.0
YAW_TOLERANCE_DEG = 2.5
CLOCK_TOLERANCE_S = 0.04


def prepare(data):
    foreign = data / 'foreign'
    if not foreign.exists():
        subprocess.run([sys.executable, str(ROOT / 'tools' / 'foreign_bags.py'), str(foreign),
                        '--only', 'a', 'c', 'd', 'e'], check=True)
    for layout in ('office', 'corridor'):
        for seed in SEEDS:
            target = data / f'{layout}-{seed}'
            if not target.exists():
                generate(target, layout=layout, seed=seed)


def tilted(rpy):
    return lambda cfg: with_rotation(cfg, rpy)


def offsets(**values):
    def apply(cfg):
        out = copy.deepcopy(cfg)
        out['header_offset_s'] = dict(values)
        return out
    return apply


def identity_rotation(cfg):
    out = copy.deepcopy(cfg)
    out['base_from_lidar_xyzw'] = [0.0, 0.0, 0.0, 1.0]
    return out


def cases(data):
    f = data / 'foreign'
    truth = {k: (f / k / 'data' / 'mapping', f / k / 'truth' / 'expected_profile.json') for k in 'acde'}
    example = ROOT / 'configs' / 'example.json'
    # name, (bag, correct profile), error, expected finding
    error_cases = [
        ('c: знак наклона перепутан (pitch −10° вместо +10°)', truth['c'], tilted([0, -20, 0]), dict(tilt=[0, 20])),
        ('c: наклон забыт (pitch 0° вместо +10°)', truth['c'], tilted([0, -10, 0]), dict(tilt=[0, 10])),
        ('c: лишний крен +8°', truth['c'], tilted([8, 0, 0]), dict(tilt=[-8, 0])),
        ('e: однолучевой, знак наклона перепутан (−6° вместо +6°)', truth['e'], tilted([0, -12, 0]),
         dict(tilt=[0, 12])),
        ('e: однолучевой, наклон забыт (0° вместо +6°)', truth['e'], tilted([0, -6, 0]), dict(tilt=[0, 6])),
        ('e: однолучевой, лишний крен −8°', truth['e'], tilted([-8, 0, 0]), dict(tilt=[8, 0])),
        ('d: лидар вверх ногами, задано единичное вращение', truth['d'], identity_rotation, dict(mirror=True)),
        ('d: сдвиг часов одометрии не учтён (истинно −0,25 с)', truth['d'], offsets(), dict(clock_odom=-0.25)),
        ('a: часы одометрии сдвинуты на +0,3 с', truth['a'], offsets(odom=0.3), dict(clock_odom=0.0)),
        ('a: часы IMU сдвинуты на −0,2 с', truth['a'], offsets(imu=-0.2), dict(clock_imu=0.0)),
        ('e: курс лидара +5°', truth['e'], tilted([0, 0, 5]), dict(yaw=-5)),
        ('a: курс лидара −8°', truth['a'], tilted([0, 0, -8]), dict(yaw=8)),
        ('a: лидар развёрнут назад (курс 180°)', truth['a'], tilted([0, 0, 180]), dict(yaw=180)),
    ]
    correct = [(f'{k}: верный профиль', truth[k]) for k in 'acde']
    correct += [(f'{layout} seed {seed}: верный профиль', (data / f'{layout}-{seed}' / 'mapping', example))
                for layout in ('office', 'corridor') for seed in SEEDS]
    return error_cases, correct


def angle_error(a, b):
    return abs((a - b + 180) % 360 - 180)


def judge(report, expected):
    """Return (found, the estimated correction as text, spurious warnings)."""
    checks = report['checks']
    expected_checks = {'tilt': 'tilt', 'mirror': 'mirror', 'yaw': 'yaw',
                       'clock_odom': 'clock_odom', 'clock_imu': 'clock_imu'}
    spurious = [w for w in report['warnings'] if w not in {expected_checks[k] for k in expected}]
    found, text = True, []
    for key, value in expected.items():
        check = checks.get(key, {})
        if key == 'tilt':
            est = check.get('correction_roll_pitch_deg', [0, 0])
            text.append(f'крен {est[0]:+.0f}°, тангаж {est[1]:+.0f}°')
            found &= check.get('status') == 'warning' and all(
                abs(e - v) <= TILT_TOLERANCE_DEG for e, v in zip(est, value))
        elif key == 'mirror':
            text.append('крен 180°' if check.get('status') == 'warning' else 'не найдено')
            found &= check.get('status') == 'warning'
        elif key == 'yaw':
            est = check.get('correction_yaw_deg', 0.0)
            text.append(f'курс {est:+.1f}°')
            found &= check.get('status') == 'warning' and angle_error(est, value) <= YAW_TOLERANCE_DEG
        else:
            est = check.get('header_offset_s')
            role = key.split('_')[1]
            text.append(f'header_offset_s.{role} = {est:+.3f} с' if est is not None else 'не найдено')
            found &= check.get('status') == 'warning' and est is not None and abs(est - value) <= CLOCK_TOLERANCE_S
    return bool(found), '; '.join(text), spurious


def expected_text(expected):
    parts = []
    for key, value in expected.items():
        if key == 'tilt':
            parts.append(f'крен {value[0]:+.0f}°, тангаж {value[1]:+.0f}°')
        elif key == 'mirror':
            parts.append('крен 180°')
        elif key == 'yaw':
            parts.append(f'курс {value:+.0f}°')
        else:
            parts.append(f'header_offset_s.{key.split("_")[1]} = {value:+.2f} с')
    return '; '.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out', type=Path)
    parser.add_argument('--budget', type=float, default=60.0)
    args = parser.parse_args()
    data = args.out / 'data'
    prepare(data)
    (args.out / 'reports').mkdir(parents=True, exist_ok=True)
    error_cases, correct = cases(data)
    rows = []
    jobs = [(n, s, e, x) for n, s, e, x in error_cases] + [(n, s, None, {}) for n, s in correct]
    for i, (name, (bag, profile), error, expected) in enumerate(jobs):
        cfg = load_config(profile)
        claimed = error(cfg) if error else cfg
        report, _ = self_check(bag, claimed, args.budget, log=lambda m: None)
        (args.out / 'reports' / f'{i:02d}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if error:
            found, estimate, spurious = judge(report, expected)
        else:
            found, estimate, spurious = None, '—', report['warnings']
        row = dict(case=name, expected=expected_text(expected) if expected else 'нет ошибки',
                   estimate=estimate, found=found, false_alarms=','.join(spurious) or '0',
                   seconds=report['seconds'], verdict=report['verdict'])
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    with (args.out / 'results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    errors = [r for r in rows if r['found'] is not None]
    clean = [r for r in rows if r['found'] is None]
    summary = dict(errors_found=sum(r['found'] for r in errors), errors=len(errors),
                   spurious_in_error_cases=sum(r['false_alarms'] != '0' for r in errors),
                   false_alarms=sum(r['verdict'] == 'warning' for r in clean), correct_runs=len(clean),
                   max_seconds=max(r['seconds'] for r in rows),
                   median_seconds=float(np.median([r['seconds'] for r in rows])))
    (args.out / 'results.json').write_text(json.dumps(dict(summary=summary, rows=rows), ensure_ascii=False,
                                                      indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
