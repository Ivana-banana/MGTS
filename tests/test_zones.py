import json

import pytest

from mgts_nav.zones import inside, load_zones, visits


ZONE = [[0, 0], [1, 0], [1, 1], [0, 1]]


def row(t, x, status='matched'):
    return dict(t=t, x=x, y=.5, status=status)


def test_entry_exit_repeat_and_boundary():
    rows = [row(i, x) for i, x in enumerate([-1, 0, .4, 2, 1, .7, 2])]
    assert visits(rows, {'room': ZONE}, 1) == [
        ('room', 1., 2., 1, 1, 2), ('room', 4., 5., 1, 1, 2),
    ]
    assert inside(1, 1, ZONE)
    assert not inside(1.001, 1, ZONE)


def test_time_gaps_and_unmatched_samples_end_visits():
    rows = [row(0, .5), row(1, .5), row(10, .5), row(11, .5, 'prediction_only'),
            row(12, .5), row(13, .5), row(14, .5, 'prediction_only'), row(15, 2)]
    assert visits(rows, {'room': ZONE}, 1) == [
        ('room', 0., 1., 0, 0, 2), ('room', 12., 13., 0, 0, 2),
    ]
    assert visits([row(0, .5), row(5, .5)], {'room': ZONE}, 1) == []
    with pytest.raises(ValueError, match='increasing'):
        visits([row(1, .5), row(1, .5)], {'room': ZONE}, 1)

@pytest.mark.parametrize('bad', [
    [[0, 0], [4, 3], [0, 3], [2, 0]],           # crossing, signed area 3
    [[0, 0], [4, 0], [4, 4], [2, 0], [0, 4]],   # vertex touches opposite edge
    [[0, 0], [4, 0], [2, 0], [4, 3], [0, 3]],   # adjacent edges retrace
    [[0, 0], [4, 0], [4, 3], [4, 0], [0, 3]],   # repeated interior vertex
])
def test_rejects_non_simple_zone(tmp_path, bad):
    path = tmp_path / 'zones.json'
    path.write_text(json.dumps(dict(frame='map', zones={'west': bad})))
    with pytest.raises(ValueError, match='west'):
        load_zones(path)


def test_closed_concave_zone_and_shared_boundary(tmp_path):
    west = [[0, 0], [1, 0], [2, 0], [2, 2], [1, 1], [0, 2], [0, 0]]
    east = [[2, 0], [3, 0], [3, 2], [2, 2]]
    path = tmp_path / 'zones.json'
    path.write_text(json.dumps(dict(frame='map', zones=dict(west=west, east=east))))
    zones = load_zones(path)
    assert inside(.5, 1, zones['west']) and not inside(1, 1.5, zones['west'])
    assert visits([row(0, 2), row(1, 2)], zones, 1) == [
        ('east', 0., 1., 0, 0, 2), ('west', 0., 1., 0, 0, 2),
    ]
