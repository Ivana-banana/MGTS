"""Register survey coordinates to a floor plan using named landmarks."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw

from .metrics import read_csv
from .slam import rotation


def coordinates(table, names):
    points = np.asarray([table[name] for name in names], dtype=float)
    if points.shape != (len(names), 2) or not np.isfinite(points).all():
        raise ValueError('Each landmark needs two finite coordinates')
    return points


def register(model, width, height, fit_scale=False):
    meter = float(model['metres_per_pixel'])
    if not np.isfinite(meter) or meter <= 0:
        raise ValueError('metres_per_pixel must be positive')
    source, target = model['map_m'], model['plan_px']
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise ValueError('map_m and plan_px must be named objects')
    if source.keys() != target.keys():
        raise ValueError(f'Unmatched landmark names: {sorted(source.keys() ^ target.keys())}')
    names = sorted(source)
    check = model.get('check', [])
    if not isinstance(check, list) or not all(isinstance(name, str) for name in check):
        raise ValueError('check must list landmark names')
    if len(set(check)) != len(check) or set(check) - set(names):
        raise ValueError('Duplicate or unknown check landmark')
    fit = np.array([name not in check for name in names], dtype=bool)
    if fit.sum() < 2:
        raise ValueError('At least two fit landmarks are required')
    xy, uv = coordinates(source, names), coordinates(target, names)
    if np.any(uv < 0) or np.any(uv > [width-1, height-1]):
        raise ValueError('Plan landmarks must lie inside the image')
    plan = np.column_stack((uv[:, 0], height-1 - uv[:, 1])) * meter
    a, b = xy[fit], plan[fit]
    ac, bc = a.mean(axis=0), b.mean(axis=0)
    a, b = a-ac, b-bc
    energies = (np.sum(a*a), np.sum(b*b))
    dot = np.sum(a*b)
    cross = np.sum(a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0])
    strength = np.hypot(dot, cross)
    if min(energies) <= 1e-16 or strength <= 1e-12 * np.sqrt(energies[0]*energies[1]):
        raise ValueError('Degenerate fit: landmarks do not determine a rotation')
    angle = np.arctan2(cross, dot)
    scale = strength / energies[0] if fit_scale else 1.
    linear = scale * rotation(angle)
    translation = bc - linear @ ac
    residuals = np.linalg.norm(xy @ linear.T + translation - plan, axis=1)
    report = dict(rotation_deg=float(np.rad2deg(angle)), scale=float(scale),
                  translation_m=translation.tolist(), metres_per_pixel=meter,
                  fit_rmse_m=float(np.sqrt(np.mean(residuals[fit]**2))),
                  check_rmse_m=float(np.sqrt(np.mean(residuals[~fit]**2))) if check else None,
                  landmarks={name: dict(role='fit' if used else 'check', residual_m=float(r))
                             for name, used, r in zip(names, fit, residuals)})
    return linear, translation, report


def overlay(image, poses, linear, translation, meter):
    route = np.asarray([(row.x, row.y) for row in poses])
    uv = (route @ linear.T + translation) / meter
    uv[:, 1] = image.height - 1 - uv[:, 1]
    points = [tuple(p) for p in uv]
    draw = ImageDraw.Draw(image)
    if len(points) > 1:
        draw.line(points, fill='white', width=7)
        draw.line(points, fill='#007187', width=4)
    for (u, v), color in [(points[0], '#147446'), (points[-1], '#b34524')]:
        draw.ellipse((u-5, v-5, u+5, v+5), fill=color, outline='white', width=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('landmarks', type=Path)
    parser.add_argument('trajectory', type=Path)
    parser.add_argument('--fit-scale', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        model = json.loads(args.landmarks.read_text(encoding='utf-8-sig'))
        with Image.open(args.plan) as source:
            image = source.convert('RGB')
        linear, translation, report = register(model, *image.size, args.fit_scale)
        overlay(image, read_csv(args.trajectory), linear, translation, report['metres_per_pixel'])
        if args.out.exists():
            raise ValueError(f'Choose a new output directory: {args.out}')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.out.parent) as tmp:
            stage = Path(tmp) / 'result'
            stage.mkdir()
            image.save(stage / 'route.png')
            (stage / 'alignment.json').write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                  encoding='utf-8')
            stage.rename(args.out)
        print(json.dumps(report, ensure_ascii=True))
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
