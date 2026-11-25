"""Register survey coordinates to a floor plan using named landmarks."""
import argparse
import json
from pathlib import Path
import sys
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


def _no_duplicates(pairs):
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f'Name {key!r} appears twice in one JSON object; a repeated landmark '
                             'would silently replace the first one')
        seen[key] = value
    return seen


def load_model(path):
    """Landmark JSON; UTF-8 with or without BOM, repeated names rejected."""
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=_no_duplicates)


def similarity(src, dst, fit_scale):
    """Least squares dst ~ scale R(angle) src + t without reflection; None if degenerate."""
    ac, bc = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - ac, dst - bc
    energies = (np.sum(a*a), np.sum(b*b))
    dot = np.sum(a*b)
    cross = np.sum(a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0])
    strength = np.hypot(dot, cross)
    if min(energies) <= 1e-16 or strength <= 1e-12 * np.sqrt(energies[0]*energies[1]):
        return None
    angle = np.arctan2(cross, dot)
    scale = strength / energies[0] if fit_scale else 1.
    linear = scale * rotation(angle)
    return angle, scale, linear, bc - linear @ ac


def consensus(src, dst, fit_scale, limit, rounds=3):
    """Largest set of fit landmarks that agree within limit on one least squares transform.

    Every pair of landmarks, and all of them together, propose a transform. Each proposal is
    refit by least squares to the landmarks it explains and the set is recounted against the
    refit, a few times (LO-RANSAC). A landmark is left out only if the fit through the agreeing
    ones misses it, not just the transform through one pair.
    A swapped or mistyped landmark spoils the least squares fit of all points, but stays outside
    the set that the correct landmarks agree on.
    """
    n = len(src)
    seeds = [tuple(range(n))] + [(i, j) for i in range(n) for j in range(i + 1, n)]
    best, best_cost = (), np.inf
    for seed in seeds:
        members, err = seed, None
        for _ in range(rounds):
            fitted = similarity(src[list(members)], dst[list(members)], fit_scale)
            if fitted is None:
                break
            err = np.linalg.norm(src @ fitted[2].T + fitted[3] - dst, axis=1)
            agreed = tuple(np.flatnonzero(err < limit))
            if agreed == members or len(agreed) < 2:
                break
            members = agreed
        if err is None:
            continue
        inliers = tuple(np.flatnonzero(err < limit))
        cost = float(np.sum(err[list(inliers)]))
        if (len(inliers), -cost) > (len(best), -best_cost):
            best, best_cost = inliers, cost
    return best


def register(model, width, height, fit_scale=False, outlier_m=0.25):
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
    fitted = similarity(xy[fit], plan[fit], fit_scale)
    if fitted is None:
        raise ValueError('Degenerate fit: landmarks do not determine a rotation')
    angle, scale, linear, translation = fitted
    residuals = np.linalg.norm(xy @ linear.T + translation - plan, axis=1)
    if not (np.isfinite(outlier_m) and outlier_m > 0):
        raise ValueError('outlier limit must be positive')
    fit_names = [name for name, used in zip(names, fit) if used]
    warnings, suspects = [], []
    if len(fit_names) >= 3:
        agree = consensus(xy[fit], plan[fit], fit_scale, outlier_m)
        agreeing = [fit_names[k] for k in agree]
        if len(agree) > len(fit_names) / 2 and len(agree) >= 3:
            suspects = [name for k, name in enumerate(fit_names) if k not in agree]
            if suspects:
                warnings.append(f'Fit landmarks {suspects} miss by more than {outlier_m:g} m the least squares '
                                f'fit through the other {len(agree)} fit landmarks, which agree with each other; '
                                'check their names and coordinates. They are still used in the fit.')
        elif len(agree) > len(fit_names) / 2:
            warnings.append(f'Only {agreeing} of {len(fit_names)} fit landmarks agree within {outlier_m:g} m; '
                            'too few to name the wrong one (at least 3 agreeing are needed). '
                            'Add landmarks or check them all.')
        else:
            warnings.append(f'No majority of fit landmarks agrees within {outlier_m:g} m (largest consistent '
                            f'set {agreeing}); check for swapped landmarks.')
    check_outliers = [name for name, used, r in zip(names, fit, residuals) if not used and r > outlier_m]
    if check_outliers:
        warnings.append(f'Check landmarks {check_outliers} miss the fit by more than {outlier_m:g} m.')
    report = dict(rotation_deg=float(np.rad2deg(angle)), scale=float(scale),
                  translation_m=translation.tolist(), metres_per_pixel=meter,
                  fit_rmse_m=float(np.sqrt(np.mean(residuals[fit]**2))),
                  check_rmse_m=float(np.sqrt(np.mean(residuals[~fit]**2))) if check else None,
                  outlier_limit_m=float(outlier_m), suspects=suspects, check_outliers=check_outliers,
                  warnings=warnings,
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
    parser.add_argument('--outlier-m', type=float, default=0.25,
                        help='landmark disagreement reported as suspect, metres on the plan (default 0.25)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        model = load_model(args.landmarks)
        with Image.open(args.plan) as source:
            image = source.convert('RGB')
        linear, translation, report = register(model, *image.size, args.fit_scale, args.outlier_m)
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
        for warning in report['warnings']:
            print(f'warning: {warning}', file=sys.stderr)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
