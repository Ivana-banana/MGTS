import pytest

from mgts_nav.quality import inspection_quality
from mgts_nav.slam import Estimate


def test_distinguishes_missing_scans_from_predictions_and_reports_boundary_losses():
    times = [0, 1, 2, 3, 10, 11, 12, 13]
    matches = {1, 4, 6}
    rows = [Estimate(t, None, 'matched' if i in matches else 'prediction_only',
                     0.8 if i in matches else 0.1) for i, t in enumerate(times)]
    report = inspection_quality(rows)
    assert report['matched_scan_fraction'] == 3 / 8
    assert report['median_inlier_fraction_on_matches'] == 0.8
    assert report['gap_threshold_s'] == 3
    assert report['scan_gaps'] == [dict(start_s=3, end_s=10, duration_s=7)]
    assert report['max_scan_interval_s'] == 7
    runs = report['prediction_runs']
    assert [(r['previous_match_s'], r['scans'], r['next_match_s']) for r in runs] == [
        (None, 1, 1), (1, 2, 10), (10, 1, 12), (12, 1, None)]
    assert (runs[1]['first_prediction_s'], runs[1]['last_prediction_s']) == (2, 3)


def test_one_scan_does_not_establish_a_cadence():
    row = Estimate(1.0, None, 'prediction_only', 0)
    report = inspection_quality([row])
    assert report['gap_threshold_s'] is None and report['max_scan_interval_s'] is None
    assert report['median_inlier_fraction_on_matches'] is None
    with pytest.raises(ValueError, match='increase'):
        inspection_quality([row, row])
