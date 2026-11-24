import pytest

from tools.evaluate_joint_outage import experiment, read_rows


def test_joint_outage_waits_for_a_new_anchor(tmp_path):
    result = experiment(tmp_path, 41)
    assert result['lidar_only']['removed'] == {'lidar': 20, 'odom': 0}
    assert result['lidar_only']['emitted_during_lidar_loss'] == 40
    for case in ('joint', 'joint_reset', 'joint_reset_recover'):
        run = result[case]
        assert run['removed'] == {'lidar': 20, 'odom': 20}
        assert run['odom_gaps'] == [[253.45, 254.5]]
        assert run['emitted_between_odom_return_and_lidar'] == 0
        assert run['emitted_during_lidar_loss'] == 10
        # The continuous stream must either end at the last pre-gap odometry or
        # resume exactly on a new matched scan (never on stale odometry).
        dense = read_rows(tmp_path/case/'continuous.csv')
        resumed = next((r for r in dense if float(r['t']) >= 254.5), None)
        if run['first_match'] is None:
            assert resumed is None
            assert run['last_output_t'] == pytest.approx(253.45)
        else:
            assert resumed['status'] == 'matched'
            assert float(resumed['t']) == run['first_match']['t']
            assert run['first_match']['xy_m'] < .25
    assert result['joint']['first_return']['status'] == 'matched'
    recovered = result['joint_reset_recover']
    assert recovered['first_match'] is not None
    assert recovered['later_max_matched_error_m'] < .25
