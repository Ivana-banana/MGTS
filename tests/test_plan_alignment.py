import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from mgts_nav.metrics import Pose
from mgts_nav.plan_alignment import overlay, register
from mgts_nav.slam import rotation


def example():
    return json.loads(Path('docs/plan-landmarks.json').read_text(encoding='utf-8-sig'))


def test_known_similarity_and_fixed_scale():
    model = example()
    with Image.open('docs/office-plan.png') as image:
        _, _, report = register(model, *image.size, fit_scale=True)
        assert register(model, *image.size)[2]['scale'] == 1
    assert report['rotation_deg'] == pytest.approx(18, abs=.15)
    assert report['scale'] == pytest.approx(1.15, abs=.002)
    np.testing.assert_allclose(report['translation_m'], [7.6, 4.1], atol=.02)
    assert report['check_rmse_m'] < .025
    edited = copy.deepcopy(model)
    edited['plan_px'][model['check'][0]][0] += 50
    result = register(edited, 900, 680, True)[2]
    assert result['translation_m'] == report['translation_m']
    assert result['check_rmse_m'] > .5


def test_bad_names_geometry_and_swap():
    model = example()
    model['map_m']['typo'] = model['map_m'].pop('door')
    with pytest.raises(ValueError, match='Unmatched.*door.*typo'):
        register(model, 900, 680)
    model = example()
    uv = model['plan_px']
    uv['door'], uv['cabinet'] = uv['cabinet'], uv['door']
    assert register(model, 900, 680, True)[2]['fit_rmse_m'] > .5
    model['map_m'] = dict.fromkeys(model['map_m'], [0, 0])
    with pytest.raises(ValueError, match='Degenerate'):
        register(model, 900, 680)


def test_swapped_landmarks_are_named_when_the_others_agree():
    model = example()
    model['check'] = []
    uv = model['plan_px']
    uv['door'], uv['cabinet'] = uv['cabinet'], uv['door']
    report = register(model, 900, 680, True)[2]
    assert sorted(report['suspects']) == ['cabinet', 'door']
    assert report['check_outliers'] == []
    clean = register(example(), 900, 680, True)[2]
    assert clean['suspects'] == [] and clean['warnings'] == []


def test_too_few_agreeing_landmarks_and_bad_check_are_warned():
    model = example()
    uv = model['plan_px']
    uv['door'], uv['cabinet'] = uv['cabinet'], uv['door']  # two of four fit landmarks
    report = register(model, 900, 680, True)[2]
    assert report['suspects'] == [] and any('No majority' in w for w in report['warnings'])
    model = example()
    model['plan_px'][model['check'][0]][0] += 50
    assert register(model, 900, 680, True)[2]['check_outliers'] == [model['check'][0]]
    with pytest.raises(ValueError, match='positive'):
        register(example(), 900, 680, True, outlier_m=0)


def test_repeated_landmark_name_is_rejected(tmp_path):
    from mgts_nav.plan_alignment import load_model
    path = tmp_path / 'landmarks.json'
    path.write_text('{"metres_per_pixel": 0.02, "map_m": {"a": [0, 0], "a": [1, 1]}}', encoding='utf-8')
    with pytest.raises(ValueError, match='twice'):
        load_model(path)
    path.write_bytes('{"metres_per_pixel": 0.02}'.encode('utf-8-sig'))
    assert load_model(path)['metres_per_pixel'] == 0.02


def noisy_clean_model(rng, count, noise_m, meter=0.02):
    xy = rng.uniform(-4, 4, (count, 2))
    plan = (xy @ rotation(np.deg2rad(18)).T) * 1.15 + [12, 10]
    uv = plan / meter
    uv[:, 1] = 999 - uv[:, 1]
    uv += rng.normal(0, noise_m / meter, uv.shape)
    names = [f'l{k}' for k in range(count)]
    return dict(metres_per_pixel=meter, map_m=dict(zip(names, xy.tolist())),
                plan_px=dict(zip(names, uv.tolist())), check=[])


@pytest.mark.parametrize('count', [3, 4, 5, 6, 8])
def test_clean_landmarks_with_marking_noise_are_not_suspects(count):
    # 0.07 m per axis, 0.1 m RMS off on the plan. The transform through one pair of landmarks
    # used to throw a correct third one past 0.25 m; the refit through the agreeing set does not.
    rng = np.random.default_rng(count)
    for _ in range(100):
        report = register(noisy_clean_model(rng, count, 0.07), 1200, 1000, True)[2]
        assert report['suspects'] == [] and report['warnings'] == []


def test_two_of_three_agreeing_is_too_few_to_name_the_wrong_one():
    model = example()
    model['check'] = []
    names = sorted(model['map_m'])[:3]
    model['map_m'] = {name: model['map_m'][name] for name in names}
    model['plan_px'] = {name: model['plan_px'][name] for name in names}
    model['plan_px'][names[0]][0] += 60
    report = register(model, 900, 680, True)[2]
    assert report['suspects'] == []
    assert any('too few to name' in w for w in report['warnings'])
    assert not any('No majority' in w for w in report['warnings'])
@pytest.mark.parametrize(('times', 'xs', 'gap', 'samples'), [
    ([0, 1, 2, 10, 20, 21, 22], [8, 18, 28, 48, 68, 78, 88], None,
     {15: True, 38: False, 48: True, 58: False, 75: True}),
    ([0, 1], [8, 88], None, {40: False}),
    ([0, 1], [8, 88], 1, {40: True}),
])
def test_overlay_preserves_time_gaps(times, xs, gap, samples):
    image = Image.new('RGB', (100, 25), 'gray')
    poses = [Pose(t, x, 12, 0) for t, x in zip(times, xs)]
    overlay(image, poses, np.eye(2), np.zeros(2), 1., gap)
    for x, connected in samples.items():
        assert image.getpixel((x, 12)) == ((0, 113, 135) if connected else (128, 128, 128))
