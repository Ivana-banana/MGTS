"""Align the survey map with an office plan from control points named on both.

Input is one JSON file, see docs/floorplan.md. Plan coordinates are converted to a
right-handed metric frame first (pixel rows grow downwards, so Y is flipped for
`"y_axis": "down"`), then a plane rotation and shift, and a scale only in
`"mode": "similarity"`, are fitted by least squares on the `fit` points. Points listed in
`check` never enter the fit and give the held-out error. Robot poses are not used.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MODES = ('rigid', 'similarity')


class AlignmentError(ValueError):
    pass


def load(path):
    path = Path(path)
    spec = json.loads(path.read_text(encoding='utf-8'))
    spec['_base'] = path.parent
    return spec


def plan_to_metric(points, plan):
    """Plan units -> right-handed plan frame in plan units scaled by units_per_meter."""
    points = np.asarray(points, dtype=float).reshape(-1, 2).copy()
    axis = plan.get('y_axis')
    if axis not in ('up', 'down'):
        raise AlignmentError('plan.y_axis must be "up" or "down" (pixel images: "down")')
    if axis == 'down':
        points[:, 1] *= -1
    upm = plan.get('units_per_meter')
    return points / upm if upm else points


def metric_to_plan(points, plan):
    points = np.asarray(points, dtype=float).reshape(-1, 2).copy()
    upm = plan.get('units_per_meter')
    if upm:
        points = points * upm
    if plan['y_axis'] == 'down':
        points[:, 1] *= -1
    return points


def umeyama(src, dst, scale, reflect=False):
    """dst ~ s R src + t; returns s, R, t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, sing, vt = np.linalg.svd(b.T @ a / len(src))
    d = np.eye(2)
    if not reflect and np.linalg.det(u @ vt) < 0:
        d[1, 1] = -1
    rot = u @ d @ vt
    var = (a ** 2).sum() / len(src)
    s = float(np.trace(np.diag(sing) @ d) / var) if scale else 1.0
    return s, rot, mu_d - s * rot @ mu_s


def apply(params, xy):
    s, rot, t = params
    return (s * (rot @ np.asarray(xy, dtype=float).T)).T + t


def _pairs(spec):
    map_pts, plan_pts = spec.get('map_points', {}), spec.get('plan_points', {})
    only_map = sorted(set(map_pts) - set(plan_pts))
    only_plan = sorted(set(plan_pts) - set(map_pts))
    if only_map or only_plan:
        raise AlignmentError(f'Names differ between map_points and plan_points: only on map {only_map}, '
                             f'only on plan {only_plan}. Check for a typo in the names.')
    check = list(spec.get('check', []))
    unknown = sorted(set(check) - set(map_pts))
    if unknown:
        raise AlignmentError(f'check lists unknown points {unknown}')
    fit = [name for name in map_pts if name not in check]
    return fit, check


def geometry(xy, tolerance):
    """Spread of the fit points: coincident -> error, collinear -> handedness unverifiable."""
    centred = xy - xy.mean(0)
    sing = np.linalg.svd(centred, compute_uv=False) / np.sqrt(len(xy))
    return dict(spread_m=float(sing[0]), width_m=float(sing[1]) if len(sing) > 1 else 0.0,
                collinear=bool(len(sing) < 2 or sing[1] < tolerance))


def align(spec, tolerance=0.05, outlier_m=None):
    """Fit and diagnostics. tolerance is the expected point picking error in metres."""
    mode = spec.get('mode', 'rigid')
    if mode not in MODES:
        raise AlignmentError(f'mode must be one of {MODES}')
    plan = spec['plan']
    if mode == 'rigid' and not plan.get('units_per_meter'):
        raise AlignmentError('rigid mode needs plan.units_per_meter; otherwise choose "similarity"')
    fit, check = _pairs(spec)
    if len(fit) < 2:
        raise AlignmentError(f'At least 2 fit points are needed, got {len(fit)}')
    src = np.array([spec['map_points'][n] for n in fit], dtype=float)
    dst = plan_to_metric([spec['plan_points'][n] for n in fit], plan)
    shape = geometry(src, tolerance)
    if shape['spread_m'] < 2 * tolerance:
        raise AlignmentError(f'Degenerate fit points: spread {shape["spread_m"]:.3f} m is within '
                             f'the picking tolerance {tolerance} m')
    params = umeyama(src, dst, scale=mode == 'similarity')
    if params[0] <= 0 or not np.isfinite(params[0]):
        raise AlignmentError('Fitted scale is not positive')
    residual = np.linalg.norm(apply(params, src) - dst, axis=1)
    # Residuals in plan-metric units; divide by scale to express them on the map in metres.
    to_m = 1.0 / params[0]
    warnings = []
    if len(fit) == 2:
        warnings.append('Two fit points determine the transform exactly; residuals are zero by construction.')
    if shape['collinear']:
        warnings.append('Fit points are collinear: a mirrored plan axis cannot be detected.')
    else:
        mirrored = umeyama(src, dst, scale=mode == 'similarity', reflect=True)
        if np.linalg.det(mirrored[1]) < 0:
            mres = np.linalg.norm(apply(mirrored, src) - dst, axis=1)
            if np.sqrt((mres ** 2).mean()) < 0.5 * np.sqrt((residual ** 2).mean()) and \
                    np.sqrt((residual ** 2).mean()) * to_m > tolerance:
                raise AlignmentError('A mirrored transform fits much better: check plan.y_axis')

    leave_one_out = {}
    if len(fit) >= 3 and not shape['collinear']:
        for i, name in enumerate(fit):
            keep = np.arange(len(fit)) != i
            if geometry(src[keep], tolerance)['spread_m'] < 2 * tolerance:
                continue
            p = umeyama(src[keep], dst[keep], scale=mode == 'similarity')
            leave_one_out[name] = float(np.linalg.norm(apply(p, src[i:i+1]) - dst[i]) / p[0])
    limit = outlier_m if outlier_m is not None else 4 * tolerance
    suspects = sorted((n for n, e in leave_one_out.items() if e > limit), key=lambda n: -leave_one_out[n])
    if suspects:
        warnings.append(f'Points {suspects} disagree with the others by more than {limit:.3g} m '
                        'when left out; check their names and coordinates.')

    rows = []
    for role, names in (('fit', fit), ('check', check)):
        for name in names:
            m = np.array(spec['map_points'][name], dtype=float)
            observed = np.array(spec['plan_points'][name], dtype=float)
            predicted = metric_to_plan(apply(params, m[None]), plan)[0]
            err = float(np.linalg.norm(apply(params, m[None])[0] - plan_to_metric(observed, plan)[0]) * to_m)
            rows.append(dict(name=name, role=role, map=m.tolist(), plan=observed.tolist(),
                             predicted_plan=predicted.tolist(), error_m=err,
                             leave_one_out_m=leave_one_out.get(name)))

    def rms(role):
        errors = [r['error_m'] for r in rows if r['role'] == role]
        return float(np.sqrt(np.mean(np.square(errors)))) if errors else None

    s, rot, t = params
    upm = plan.get('units_per_meter')
    return dict(
        mode=mode, fit_points=len(fit), check_points=len(check),
        rotation_deg=float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))),
        scale=float(s) if mode == 'similarity' else 1.0,
        plan_units_per_map_meter=float(s * (upm or 1.0)) if mode == 'similarity' else float(upm),
        translation=t.tolist(),
        translation_units='metres of the plan frame' if upm else 'plan units, Y up',
        fit_rms_m=rms('fit'), check_rms_m=rms('check'),
        check_max_m=max((r['error_m'] for r in rows if r['role'] == 'check'), default=None),
        geometry=shape, warnings=warnings, suspects=suspects, points=rows, _params=params)


def read_route(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return np.array([[float(r['x']), float(r['y'])] for r in csv.DictReader(f)])


def overlay(spec, report, output, route=None):
    """Draw route and control points on the plan image (plan units must be pixels)."""
    plan = spec['plan']
    image_path = spec['_base'] / plan['image']
    with Image.open(image_path) as src:
        image = src.convert('RGB')
    draw = ImageDraw.Draw(image)
    if plan['y_axis'] != 'down':
        raise AlignmentError('Overlay needs a pixel plan with y_axis "down"')
    params = report['_params']
    if route is not None and len(route):
        uv = metric_to_plan(apply(params, route), plan)
        draw.line([tuple(p) for p in uv], fill=(220, 90, 0), width=3)
        draw.ellipse([*(uv[0] - 6), *(uv[0] + 6)], fill=(0, 160, 0))
        draw.ellipse([*(uv[-1] - 6), *(uv[-1] + 6)], fill=(200, 0, 0))
    for row in report['points']:
        p, q = np.array(row['plan']), np.array(row['predicted_plan'])
        colour = (0, 120, 220) if row['role'] == 'fit' else (150, 0, 180)
        if row['role'] == 'fit':
            draw.ellipse([*(p - 6), *(p + 6)], outline=colour, width=2)
        else:
            draw.rectangle([*(p - 6), *(p + 6)], outline=colour, width=2)
        draw.line([tuple(p), tuple(q)], fill=(220, 0, 0), width=2)
        draw.text(tuple(p + 8), f"{row['name']} {row['error_m']:.2f} m", fill=colour)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def public(report):
    return {k: v for k, v in report.items() if not k.startswith('_')}


def synthetic(output, rotation_deg=17.0, pixels_per_meter=48.0, margin_px=70,
              noise_m=0.02, seed=5):
    """A plan of the demo office drawn with a known transform, plus a point file."""
    from .demo import WALLS, pose_at

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    a = np.radians(rotation_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    rotated = (pixels_per_meter * (rot @ WALLS.reshape(-1, 2).T)).T
    # Pixel u = x' + shift_u, v = shift_v - y' keeps the whole drawing on the canvas.
    shift_px = (margin_px - rotated[:, 0].min(), margin_px + rotated[:, 1].max())
    size = (int(np.ptp(rotated[:, 0]) + 2 * margin_px), int(np.ptp(rotated[:, 1]) + 2 * margin_px))

    def to_px(xy):
        p = (pixels_per_meter * (rot @ np.asarray(xy, float).T)).T
        return np.column_stack((p[:, 0] + shift_px[0], shift_px[1] - p[:, 1]))

    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    for seg in WALLS:
        draw.line([tuple(p) for p in to_px(seg)], fill=(0, 0, 0), width=3)
    image.save(output / 'plan.png')
    corners = {'NW': (-4.2, 4.8), 'SW': (-4.2, -2.0), 'SE': (5.2, -2.0), 'NE': (5.2, 4.8),
               'cabinet': (2.8, 2.6), 'pillar': (-2.8, -0.5), 'desk': (1.3, 3.3), 'door': (3.8, -1.4)}
    plan_points = {n: [round(v, 1) for v in to_px([xy])[0]] for n, xy in corners.items()}
    map_points = {n: (np.array(xy) + rng.normal(0, noise_m, 2)).round(3).tolist() for n, xy in corners.items()}
    spec = dict(mode='similarity', plan=dict(image='plan.png', y_axis='down', units='px'),
                map_points=map_points, plan_points=plan_points, check=['cabinet', 'desk', 'door'])
    (output / 'points.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
    times = np.linspace(0, 10, 101)
    with (output / 'route.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'x', 'y', 'yaw'])
        for t in times:
            writer.writerow([t, *pose_at('inspection', t)])
    truth = dict(rotation_deg=rotation_deg, pixels_per_meter=pixels_per_meter, shift_px=[float(v) for v in shift_px],
                 expected_translation_plan_units_y_up=[float(shift_px[0]), -float(shift_px[1])],
                 point_noise_m=noise_m, seed=seed)
    (output / 'plan_truth.json').write_text(json.dumps(truth, indent=2), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('points', type=Path, nargs='?', help='control point JSON')
    parser.add_argument('--route', type=Path, help='trajectory.csv in the map frame to draw on the plan')
    parser.add_argument('--out', type=Path, help='directory for alignment.json and overlay.png')
    parser.add_argument('--tolerance', type=float, default=0.05, help='point picking error, m')
    parser.add_argument('--outlier', type=float, help='leave-one-out limit, m (default 4x tolerance)')
    parser.add_argument('--synthetic', type=Path, metavar='OUT',
                        help='write a synthetic plan with known transform and align it')
    args = parser.parse_args()
    try:
        if args.synthetic is not None:
            synthetic(args.synthetic)
            args.points = args.synthetic / 'points.json'
            args.route = args.route or args.synthetic / 'route.csv'
            args.out = args.out or args.synthetic
        if args.points is None:
            parser.error('give a control point JSON or --synthetic OUT')
        spec = load(args.points)
        report = align(spec, args.tolerance, args.outlier)
        out = args.out or args.points.parent
        out.mkdir(parents=True, exist_ok=True)
        text = json.dumps(public(report), indent=2, ensure_ascii=False)
        (out / 'alignment.json').write_text(text, encoding='utf-8')
        if spec['plan'].get('image'):
            overlay(spec, report, out / 'overlay.png', read_route(args.route) if args.route else None)
        print(text)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
