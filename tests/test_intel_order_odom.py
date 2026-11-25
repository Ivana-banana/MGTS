import numpy as np
from tools.intel_order_odom import wheel_coverage


def test_bracketed_scan_endpoints_can_hide_a_wheel_gap():
    rows = [(0, None), (1_000_000_000, None), (1_400_000_000, None)]
    wheel = np.array([0., .1, .9, 1., 1.4])
    result = wheel_coverage(rows, wheel, .5)
    assert result["wheel_covered"] == 3
    assert [item["wheel"] for item in result["breaks"]] == [False, True]
