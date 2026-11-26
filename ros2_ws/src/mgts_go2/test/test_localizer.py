import math

import numpy as np

from mgts_go2.core import cloud, se3
from mgts_go2.core.icp import PointMap
from mgts_go2.core.localizer import LOST, MATCHED, Localizer, LocalizerParams

from synthetic import base_pose, room, sweep


def _route(n=60):
    """Base poses along an arc inside the room (ground truth in the map frame)."""
    out = []
    for k in range(n):
        a = k / n
        out.append(base_pose(3.0 + 4.0 * a, 2.5 + 3.0 * a * a, 0.3 + 1.2 * a))
    return out


def _setup():
    world = room()
    target = PointMap(cloud.voxel_downsample(world, 0.05))
    return world, target


def test_tracks_despite_drifting_odometry():
    world, target = _setup()
    loc = Localizer(target, LocalizerParams())
    truth = _route()
    errs = []
    for k, t_true in enumerate(truth):
        # odometry drifts: 0.3 deg and 1 cm per sweep, i.e. 18 deg and 0.6 m over the run
        drift = se3.make(se3.rot_z(math.radians(0.3) * k), (0.01 * k, 0.0, 0.0))
        t_odom = drift @ t_true
        if k == 0:
            loc.set_map_odom(se3.inv(drift))
        r = loc.step(float(k), sweep(world, t_true, seed=k), t_odom)
        dt, dr = se3.delta_norm(se3.inv(t_true) @ r.t_map_base)
        errs.append((dt, dr, r.status))
    assert all(s == MATCHED for _, _, s in errs)
    assert max(e[0] for e in errs) < 0.05
    assert max(e[1] for e in errs) < math.radians(1.0)
    odom_err = se3.delta_norm(se3.inv(truth[-1]) @ (se3.make(se3.rot_z(math.radians(0.3) * 59), (0.59, 0, 0)) @ truth[-1]))
    assert odom_err[0] > 0.5  # the corrupted odometry alone would be off by more than half a metre


def test_wrong_start_is_corrected():
    world, target = _setup()
    loc = Localizer(target, LocalizerParams())
    truth = _route()
    loc.set_map_odom(se3.make(se3.rot_z(math.radians(12)), (0.6, -0.5, 0.0)))  # wrong external start
    for k, t_true in enumerate(truth[:12]):
        r = loc.step(float(k), sweep(world, t_true, seed=k), t_true)
    dt, dr = se3.delta_norm(se3.inv(truth[11]) @ r.t_map_base)
    assert r.status == MATCHED and dt < 0.05 and dr < math.radians(1.0)


def test_local_search_recovers_offset_beyond_icp_basin():
    world, target = _setup()
    loc = Localizer(target, LocalizerParams())
    t_true = base_pose(5.0, 4.0, 0.8)
    wrong = se3.make(se3.rot_z(math.radians(-17)), (0.9, -0.8, 0.0))
    # prediction off by more than a metre and 17 deg around the robot
    pivot = se3.make(None, t_true[:3, 3]) @ wrong @ se3.make(None, -t_true[:3, 3])
    loc.set_map_odom(pivot)
    src = loc.prepare(sweep(world, t_true, n=1200, seed=7))
    corr, best, second = loc.local_search(src, t_true)
    assert corr is not None, (best, second)
    dt, dr = se3.delta_norm(se3.inv(t_true) @ (corr @ t_true))
    assert dt < 0.05 and dr < math.radians(1.0)


def test_global_search_finds_start_without_prior_knowledge():
    world, target = _setup()
    places = [base_pose(x, y, 0.0) for x in np.arange(1.5, 9.0, 1.0) for y in np.arange(1.5, 7.0, 1.0)]
    loc = Localizer(target, LocalizerParams(), places)
    t_true = base_pose(6.2, 3.1, 2.4)
    odom = base_pose(0.0, 0.0, 0.0)  # odometry frame unrelated to the map
    r = loc.step(0.0, sweep(world, t_true, n=1200, seed=5), odom)
    assert r.status == MATCHED
    dt, dr = se3.delta_norm(se3.inv(t_true) @ r.t_map_base)
    assert dt < 0.05 and dr < math.radians(1.0)
