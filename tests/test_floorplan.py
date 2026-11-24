import json

import numpy as np
import pytest

from mgts_nav import floorplan
from mgts_nav.floorplan import AlignmentError, align


def make_spec(map_points, deg=-25.0, ppm=40.0, shift=(500.0, 300.0), mode='similarity', check=(), **plan):
    a = np.radians(deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    plan_points = {}
    for name, xy in map_points.items():
        p = ppm * rot @ np.asarray(xy, float)
        plan_points[name] = [p[0] + shift[0], shift[1] - p[1]]  # pixel rows grow downwards
    return dict(mode=mode, plan=dict(y_axis='down', **plan), map_points=dict(map_points),
                plan_points=plan_points, check=list(check))


SQUARE = {'a': (0, 0), 'b': (6, 0), 'c': (6, 4), 'd': (0, 4), 'e': (3, 1), 'f': (1, 3), 'g': (5, 2)}


def test_recovers_rotation_scale_shift_and_held_out_error():
    report = align(make_spec(SQUARE, check=['f', 'g']))
    assert report['rotation_deg'] == pytest.approx(-25.0)
    assert report['plan_units_per_map_meter'] == pytest.approx(40.0)
    assert report['translation'] == pytest.approx([500.0, -300.0])
    assert report['check_points'] == 2 and report['check_rms_m'] < 1e-9
    assert not report['warnings']


def test_noisy_picks_give_held_out_error_near_the_noise():
    rng = np.random.default_rng(3)
    spec = make_spec(SQUARE, check=['f', 'g'])
    spec['map_points'] = {n: (np.array(xy) + rng.normal(0, 0.02, 2)).tolist() for n, xy in SQUARE.items()}
    report = align(spec)
    assert report['rotation_deg'] == pytest.approx(-25.0, abs=0.5)
    assert 0 < report['check_max_m'] < 0.1


def test_rigid_mode_uses_declared_plan_scale():
    spec = make_spec(SQUARE, ppm=20.0, mode='rigid', units_per_meter=20.0)
    report = align(spec)
    assert report['scale'] == 1.0 and report['fit_rms_m'] < 1e-9
    spec['plan']['units_per_meter'] = 25.0  # wrong declared scale shows up as residuals
    assert align(spec)['fit_rms_m'] > 0.3
    del spec['plan']['units_per_meter']
    with pytest.raises(AlignmentError, match='units_per_meter'):
        align(spec)


def test_coincident_points_are_rejected_and_collinear_points_warned():
    with pytest.raises(AlignmentError, match='Degenerate'):
        align(make_spec({'a': (1, 1), 'b': (1.01, 1), 'c': (1, 1.01)}))
    report = align(make_spec({'a': (0, 0), 'b': (2, 0), 'c': (5, 0)}))
    assert any('collinear' in w for w in report['warnings'])
    two = align(make_spec({'a': (0, 0), 'b': (2, 1)}))
    assert any('Two fit points' in w for w in two['warnings'])


def test_name_typo_and_swapped_points_are_reported():
    spec = make_spec(SQUARE)
    spec['plan_points']['dd'] = spec['plan_points'].pop('d')
    with pytest.raises(AlignmentError, match=r"only on map \['d'\].*only on plan \['dd'\]"):
        align(spec)
    spec = make_spec(SQUARE, check=['g'])
    p = spec['plan_points']
    p['b'], p['c'] = p['c'], p['b']  # labels of two neighbouring corners swapped
    report = align(spec)
    assert sorted(report['suspects']) == ['b', 'c']
    assert any('disagree' in w for w in report['warnings'])


def test_swapped_diagonal_points_are_not_taken_for_a_mirrored_axis(tmp_path):
    spec = floorplan.load(floorplan.synthetic(tmp_path / 'fp') / 'points.json')
    p = spec['plan_points']
    p['NW'], p['SE'] = p['SE'], p['NW']
    report = align(spec)
    assert sorted(report['suspects']) == ['NW', 'SE']
    assert not any('y_axis' in w for w in report['warnings'])


def test_mistakes_among_check_points_are_reported():
    spec = make_spec(SQUARE, check=['f', 'g'])
    p = spec['plan_points']
    p['f'], p['g'] = p['g'], p['f']
    report = align(spec)
    assert report['check_outliers'] == ['f', 'g'] and report['suspects'] == []


def test_input_file_errors(tmp_path):
    spec = make_spec(SQUARE)
    text = json.dumps({k: v for k, v in spec.items()})
    path = tmp_path / 'bom.json'
    path.write_bytes(text.encode('utf-8-sig'))
    assert floorplan.load(path)['mode'] == 'similarity'
    path.write_text('{"map_points": {"a": [0, 0], "a": [1, 1]}}', encoding='utf-8')
    with pytest.raises(AlignmentError, match='twice'):
        floorplan.load(path)
    spec['plan']['units_per_meter'] = -20
    with pytest.raises(AlignmentError, match='positive'):
        align(spec)
    spec = make_spec(SQUARE)
    spec['plan_points'] = {n: [10.0, 10.0] for n in spec['plan_points']}
    with pytest.raises(AlignmentError, match='coincide on the plan'):
        align(spec)


def test_wrong_y_axis_is_detected():
    spec = make_spec(SQUARE)
    spec['plan']['y_axis'] = 'up'
    with pytest.raises(AlignmentError, match='mirrored'):
        align(spec)


def test_synthetic_command_writes_overlay(tmp_path):
    out = floorplan.synthetic(tmp_path / 'fp')
    spec = floorplan.load(out / 'points.json')
    report = align(spec)
    truth = json.loads((out / 'plan_truth.json').read_text(encoding='utf-8'))
    assert report['rotation_deg'] == pytest.approx(truth['rotation_deg'], abs=0.5)
    assert report['plan_units_per_map_meter'] == pytest.approx(truth['pixels_per_meter'], rel=0.01)
    assert report['check_max_m'] < 0.1
    assert report['translation'] == pytest.approx(truth['expected_translation_plan_units_y_up'], abs=2.0)
    assert report['suspects'] == [] and report['warnings'] == []
    with pytest.raises(FileExistsError, match='new output'):
        floorplan.synthetic(out)
    floorplan.overlay(spec, report, out / 'overlay.png', floorplan.read_route(out / 'route.csv'))
    assert (out / 'overlay.png').stat().st_size > 0
