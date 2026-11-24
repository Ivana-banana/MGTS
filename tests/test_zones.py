import pytest

from mgts_nav.zones import inside, visits


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
