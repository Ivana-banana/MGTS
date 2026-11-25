from tools.intel_laser_clock import analyze


def test_stable_sort_recovers_late_scans_and_keeps_first_measurement():
    def row(t, value):
        return ["FLASER", "180", *([value] * 180), *(["0"] * 6), t, "host", t]

    rows = [row("0", "1"), row("0.5", "2"), row("0.2", "3"), row("0.2", "4"),
            row("0.5", "2.0"), row("1", "5")]
    report, control, ordered = analyze(rows)
    assert [r for _, r in control] == [rows[0], rows[1], rows[5]]
    assert [r for _, r in ordered] == [rows[0], rows[2], rows[1], rows[5]]
    assert report["duplicate_stamps"] == 2 and report["conflicting_groups"] == 1
    assert report["adjacent_inversions"] == 1 and report["max_reorder_lag_s"] == .3
    assert report["control"]["covered_fraction"] == 0
    assert report["sorted"]["covered_fraction"] == .5
