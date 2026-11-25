import pytest

from tools.heap1_warning_score import score, warning


def test_threshold_and_missing_support():
    assert warning('matched', '.549999') is True
    assert warning('matched', '.55') is False
    assert warning('prediction_only', '.3') is None
    assert warning('matched', '') is None
    for value in ('nan', 'inf', '-.01', '1.001'):
        with pytest.raises(ValueError):
            warning('matched', value)


def test_scoring_without_extrapolation(tmp_path):
    truth = tmp_path / 'truth.csv'
    truth.write_text('t,x,y,yaw\n' + ''.join(f'{t},0,0,0\n' for t in (1, 2, 3, 10, 11)))
    path = tmp_path / 'poses.csv'
    path.write_text('t,x,y,yaw,status,inlier_fraction\n'
                    '0,0,0,0,matched,.4\n'
                    '1,.2,0,0,matched,.54\n'
                    '1.5,.21,0,0,matched,.54\n'
                    '2,.21,0,0,matched,.55\n'
                    '3,0,0,0,matched,.55\n'
                    '5,0,0,0,matched,.4\n'
                    '10,0,0,0,prediction_only,0\n'
                    '11,1,0,0,matched,\n')
    result = score(path, truth, 10)
    assert result['warning'] == dict(tp=1, fn=1, fp=1, tn=1, sensitivity=.5,
                                     false_alarm_rate=.5, coverage=.4)
    assert result['baseline']['fn'] == 3 and result['baseline_same_support']['fn'] == 2
    assert result['matched_missing_truth'] == 2 and result['matched_missing_support'] == 1
    assert result['missing_scans'] == 2 and result['other_status'] == 1
