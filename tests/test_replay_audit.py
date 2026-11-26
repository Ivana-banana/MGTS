import json
import subprocess
import sys

import numpy as np
import pytest
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore


@pytest.fixture
def source(tmp_path):
    store = get_typestore(Stores.ROS2_HUMBLE)
    types = store.types
    cloud_type = 'sensor_msgs/msg/PointCloud2'
    path = tmp_path / 'source'
    with Writer(path, version=8) as writer:
        connection = writer.add_connection('/utlidar/cloud_base', cloud_type, typestore=store)
        for second in (10, 11):
            header = types['std_msgs/msg/Header'](types['builtin_interfaces/msg/Time'](second, 0), 'base_link')
            cloud = types[cloud_type](header, 1, 0, [], False, 0, 0, np.empty(0, np.uint8), True)
            writer.write(connection, second * 1_000_000_000 + 5, store.serialize_cdr(cloud, cloud_type))
    return path


@pytest.mark.parametrize(('second_row', 'missing', 'extra', 'code'), [
    ('11,missing_odometry,400,0', [], [], 0),
    ('10,matched,12,0', [11.], [10.], 1),
    ('11,unprocessed,400,1', [], [], 1),
])
def test_audit_reconciles_headers_and_reports_replay_loss(source, tmp_path, second_row, missing, extra, code):
    timings = tmp_path / 'timings.csv'
    timings.write_text(f'header_s,status,latency_ms,pending\n10,matched,12,0\n{second_row}\n', encoding='utf-8')
    report = subprocess.run([sys.executable, '-m', 'mgts_nav.replay_audit', str(source), str(timings)],
                            capture_output=True, text=True)
    assert report.returncode == code, report.stderr
    summary = json.loads(report.stdout)
    assert summary['missing_headers_s'] == missing
    assert summary['extra_headers_s'] == extra
    assert summary['pose_latency_ms']['p95'] == 12.
