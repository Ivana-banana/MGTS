import numpy as np
from tools.evaluate_visit_agreement import confirmed


def test_one_same_zone_interval_must_contain_two_samples_and_half_the_visit():
    times = np.arange(10)
    visit = ("zone", 1, 6, 1, 1, 6)
    short = ("zone", 1, 2, 1, 1, 2)
    assert confirmed([visit], [short, ("zone", 5, 6, 1, 1, 2)], times) == []
    assert confirmed([visit], [("other", 1, 6, 1, 1, 6)], times) == []
    assert confirmed([visit], [("zone", 4, 6, 1, 1, 3)], times) == [visit]
    assert confirmed([short], [("zone", 2, 3, 1, 1, 2)], times) == []
