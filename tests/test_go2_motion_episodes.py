import numpy as np

from tools.go2_motion_episodes import pick, runs, select


def test_runs_are_inclusive():
    assert runs(np.array([0, 1, 1, 0, 1, 0, 0, 1], bool)) == [(1, 2), (4, 4), (7, 7)]


def test_pick_keeps_each_direction_and_drops_short_or_weak_runs():
    signal = np.zeros(40)
    signal[2:6] = [50, 100, 120, 60]      # strong left, 4 samples
    signal[10:12] = [200, 200]            # strongest, but only 2 samples
    signal[15:19] = [50, 80, 85, 50]      # peak below threshold
    signal[22:26] = [-60, -150, -95, -50]  # strong right
    signal[30:34] = [-60, -95, -91, -50]  # weaker right
    assert pick(signal, 45, 90, top=1) == [(2, 5, 4), (22, 25, 23)]
    assert pick(signal, 45, 90, top=2) == [(2, 5, 4), (22, 25, 23), (30, 33, 31)]


def test_select_finds_turn_and_braking_and_keeps_control_apart():
    t = np.arange(0, 30, 0.07)
    speed = np.where(t < 20, 0.8, np.maximum(0.0, 0.8 - 3.0 * (t - 20)))
    x = np.cumsum(speed) * 0.07
    yaw = np.where(t < 10, 0.0, np.where(t < 11, np.radians(120) * (t - 10), np.radians(120)))
    xy = np.column_stack([x, np.zeros_like(x)])
    episodes, control = select(t, xy, yaw)
    kinds = {(e['kind'], e['direction']) for e in episodes}
    assert ('turn', 'left') in kinds and ('acceleration', 'braking') in kinds
    used = {i for e in episodes for i in e['scans']}
    assert len(control) and not used & set(control)
    assert all(abs(t[i] - 10.5) > 1.0 and abs(t[i] - 20.1) > 0.5 for i in control)
