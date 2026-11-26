import csv
from collections import Counter

from tools import go2_subscriber_audit as observer


def test_reordered_callbacks_and_missing_pose(tmp_path, monkeypatch):
    stamps = Counter([10, 11])  # the first source scan has no published pose
    monkeypatch.setattr(observer, 'stamps', lambda path, topic, msgtype:
                        stamps if topic == '/utlidar/cloud_base' else Counter([11]))
    monkeypatch.setattr(observer, 'callback_times', lambda folder: None)
    with (tmp_path/'observations.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['topic', 'sec', 'nanosec', 'callback_monotonic_ns'])
        writer.writerows([('scan', 0, 10, 1000000), ('pose', 0, 11, 2000000),
                          ('scan', 0, 11, 3000000), ('scan', 0, 11, 3500000)])
    result = observer.audit(tmp_path, tmp_path, tmp_path)
    assert result['negative'] == result['matched'] == 1
    assert result['subscriber_delta_ms']['p50'] == -1
    assert result['unpaired_scan_stamps'] == [10]
    assert result['duplicate_observations'] == {'scan': 1}
    assert not result['changed_release_stamps']
