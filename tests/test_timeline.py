import csv

from PIL import Image
import pytest

from mgts_nav import timeline


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'x', 'y', 'yaw', 'status', 'inlier_fraction'])
        for t, status, fraction in rows:
            writer.writerow([t, 0, 0, 0, status, fraction])
    return path


def test_intervals_name_prediction_runs_and_gaps_with_csv_lines(tmp_path):
    rows = [(100.0, 'matched', 0.9), (100.1, 'prediction_only', 0.1), (100.2, 'prediction_only', 0.2),
            (100.3, 'matched', 0.8), (101.3, 'matched', 0.8), (101.4, 'matched', 0.7)]
    path = write_csv(tmp_path / 'trajectory.csv', rows)
    found = timeline.plot(path, tmp_path / 'timeline.png')
    assert [(i['label'], i['first_line'], i['last_line'], i['scans']) for i in found] == [
        ('P1', 3, 4, 2), ('G1', 5, 6, 0)]
    assert found[0]['start_rel_s'] == pytest.approx(0.1)
    assert found[1]['end_rel_s'] == pytest.approx(1.3)
    with (tmp_path / 'timeline.csv').open(encoding='utf-8') as f:
        table = list(csv.DictReader(f))
    assert [r['label'] for r in table] == ['P1', 'G1']
    # The listed lines really hold the interval boundaries in the source CSV.
    lines = path.read_text(encoding='utf-8').splitlines()
    assert lines[int(table[0]['first_line']) - 1].startswith('100.1,')


def test_matched_points_are_not_joined_across_a_gap(tmp_path):
    rows = [(t / 10, 'matched', 0.5) for t in range(10)] + [(t / 10, 'matched', 0.5) for t in range(40, 50)]
    path = write_csv(tmp_path / 'trajectory.csv', rows)
    timeline.plot(path, tmp_path / 'gap.png')
    image = Image.open(tmp_path / 'gap.png').convert('RGB')
    width, height = image.size
    x0, x1 = timeline.LEFT, width - timeline.RIGHT
    y = round(timeline.TOP + 0.5 * (height - timeline.BOTTOM - timeline.TOP))
    mid_gap = round(x0 + (2.5 - 0) / 4.9 * (x1 - x0))
    column = [image.getpixel((mid_gap, yy)) for yy in range(y - 3, y + 4)]
    assert timeline.BLUE not in column
    before = round(x0 + 0.45 / 4.9 * (x1 - x0))
    assert timeline.BLUE in [image.getpixel((before, yy)) for yy in range(y - 3, y + 4)]


def test_single_scan_exports_and_window_selects_lines(tmp_path):
    path = write_csv(tmp_path / 'one.csv', [(5.0, 'prediction_only', 0.0)])
    assert [i['label'] for i in timeline.plot(path, tmp_path / 'one.png')] == ['P1']
    assert Image.open(tmp_path / 'one.png').size == timeline.SIZE

    rows = [(t / 10, 'matched' if t % 4 else 'prediction_only', 0.6) for t in range(30)]
    path = write_csv(tmp_path / 'long.csv', rows)
    found = timeline.plot(path, tmp_path / 'window.png', window=(1.0, 1.25))
    assert [(i['first_line'], i['last_line']) for i in found] == [(14, 14)]
    with pytest.raises(ValueError, match='No scans'):
        timeline.plot(path, tmp_path / 'none.png', window=(10, 11))


def test_rejects_trajectory_without_status(tmp_path):
    path = tmp_path / 'baseline.csv'
    path.write_text('t,x,y,yaw\n0,0,0,0\n', encoding='utf-8')
    with pytest.raises(ValueError, match='status'):
        timeline.plot(path, tmp_path / 'x.png')


def test_window_keeps_time_from_first_scan_of_the_record(tmp_path):
    rows = [(10 + t / 10, 'matched' if t != 25 else 'prediction_only', 0.6) for t in range(40)]
    path = write_csv(tmp_path / 'trajectory.csv', rows)
    found = timeline.plot(path, tmp_path / 'w.png', window=(2.0, 3.0))
    assert found[0]['start_rel_s'] == pytest.approx(2.5)
