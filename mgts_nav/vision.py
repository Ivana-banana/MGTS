"""Проверка поворота на месте по двум кадрам. Запуск: python -m mgts_nav.vision PAIR.json

PAIR.json — наш промежуточный формат. Официальная схема треков пока неизвестна.
Нужны матрица K (3x3 для обоих кадров), base_from_camera_xyzw (поворот
из оптического кадра в базу), tracks (строки [id, u0, v0, u1, v1]).
Координаты — ректифицированные пиксели с одной калибровкой, ID уникальны
внутри пары. Оптические оси: x вправо, y вниз, z вперёд. База: x вперёд,
y влево, z вверх. Выход yaw_rad положителен при левом повороте базы.

Два обязательных для принятия внешних ограничения: camera_motion_bound_m —
верхняя граница перемещения оптического центра между кадрами с учётом
смещения камеры от оси вращения и ошибки кинематики; min_scene_range_m —
нижняя граница дальности статических ориентиров от первой позиции камеры.
При отсутствии достоверных границ оставьте null. Иначе параллакс на плоской
стене может выглядеть как поворот; одни и те же треки не подтверждают границы.
Для опорных примеров треки и границы задаёт генератор в tests/test_vision.py.

--angle-deg задаёт допуск ошибки направления в градусах (по умолчанию 0.7).
Возможный параллакс должен укладываться в половину допуска. Для оценки
нужны ≥12 широко распределённых статических треков, согласие ≥80% и
поворот базы без значительного наклона (≤3°). Результат относится к одной
паре кадров; перенос и метрическая скорость по нему не восстанавливаются.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _bearings(pixels, intrinsics):
    homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
    rays = np.linalg.solve(intrinsics, homogeneous.T).T
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _rotation(first, second):
    u, _, vt = np.linalg.svd(second.T @ first)
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    return u @ vt


def _angular_errors(first, second, rotation):
    predicted = first @ rotation.T
    return np.arctan2(np.linalg.norm(np.cross(predicted, second), axis=1),
                      np.sum(predicted * second, axis=1))


def _spread(rays):
    # Collinear image points project into a plane through the optical centre.
    return np.linalg.svd(rays, compute_uv=False)[-1] / np.sqrt(len(rays))


def estimate(pair, angle_deg=0.7):
    """Возвращает JSON-совместимый результат. Отказ содержит причину и yaw_rad=null."""
    if not np.isfinite(angle_deg) or not 0 < angle_deg <= 5:
        raise ValueError('angle_deg must be in (0, 5]')
    rows = pair['tracks']
    if any(len(row) != 5 for row in rows):
        raise ValueError('Each track must be [id, u0, v0, u1, v1]')
    ids = [row[0] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Track IDs must be unique')
    result = dict(accepted=False, yaw_rad=None, tracks=len(rows), inliers=0)
    if len(rows) < 12:
        return dict(result, reason='too_few_tracks')
    pixels = np.asarray([row[1:] for row in rows], dtype=float)
    k = np.asarray(pair['K'], dtype=float)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1]) or not np.isclose(k[1, 0], 0)):
        raise ValueError('Expected finite camera intrinsics with positive focal lengths')
    if not np.isfinite(pixels).all():
        raise ValueError('Track coordinates must be finite')
    quat = np.asarray(pair['base_from_camera_xyzw'], dtype=float)
    if quat.shape != (4,) or not np.isfinite(quat).all() or abs(np.linalg.norm(quat) - 1) > .001:
        raise ValueError('Expected a unit quaternion base_from_camera_xyzw')
    mount = Rotation.from_quat(quat).as_matrix()
    motion, distance = pair.get('camera_motion_bound_m'), pair.get('min_scene_range_m')
    if motion is None or distance is None:
        return dict(result, reason='missing_motion_or_range_bound')
    if not np.isfinite([motion, distance]).all() or motion < 0 or distance <= 0:
        raise ValueError('Expected finite nonnegative motion bound and positive minimum range')
    gate = np.radians(angle_deg)
    if motion / distance > np.sin(gate / 2):
        return dict(result, reason='possible_parallax')
    first, second = _bearings(pixels[:, :2], k), _bearings(pixels[:, 2:], k)
    if min(_spread(first), _spread(second)) < .04:
        return dict(result, reason='insufficient_angular_spread')
    best_mask = np.zeros(len(rows), dtype=bool)
    best_score = (0, -np.inf)
    rng = np.random.default_rng(42)
    for _ in range(80):
        subset = rng.choice(len(rows), size=3, replace=False)
        fitted = _rotation(first[subset], second[subset])
        errors = _angular_errors(first, second, fitted)
        mask = errors <= gate
        if mask.any():
            score = (int(mask.sum()), -np.median(errors[mask]))
            if score > best_score:
                best_mask, best_score = mask, score
    min_count = max(12, int(np.ceil(.8 * len(rows))))
    result['inliers'] = int(best_mask.sum())
    if result['inliers'] < min_count:
        return dict(result, reason='weak_rotation_consensus')
    fitted = _rotation(first[best_mask], second[best_mask])
    errors = _angular_errors(first, second, fitted)
    kept = errors <= gate
    result['inliers'] = int(kept.sum())
    if result['inliers'] < min_count:
        return dict(result, reason='weak_rotation_consensus')
    if min(_spread(first[kept]), _spread(second[kept])) < .04:
        return dict(result, reason='insufficient_angular_spread')
    # Directions in a static scene rotate opposite to the moving camera.
    base_delta = mount @ fitted.T @ mount.T
    tilt = np.arccos(np.clip(base_delta[2, 2], -1, 1))
    if tilt > np.radians(3):
        return dict(result, reason='nonplanar_rotation')
    yaw = np.arctan2(base_delta[1, 0], base_delta[0, 0])
    return dict(result, accepted=True, reason='accepted', yaw_rad=float(yaw),
                residual_p90_deg=float(np.degrees(np.percentile(errors[kept], 90))))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pair', type=Path)
    parser.add_argument('--angle-deg', type=float, default=.7)
    args = parser.parse_args()
    try:
        pair = json.loads(args.pair.read_text(encoding='utf-8-sig'))
        result = estimate(pair, args.angle_deg)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
