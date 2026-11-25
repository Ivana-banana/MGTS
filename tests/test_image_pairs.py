import json
from io import BytesIO
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.transform import Rotation

from mgts_nav.camera_demo import BASE_FROM_CAMERA
from mgts_nav.image_pairs import compressed_gray, features, gray, tracks
from mgts_nav.vision import estimate


@pytest.mark.parametrize('dtype', ['<u2', '>u2'])
def test_mono16_intensity_with_odd_stride(dtype):
    samples = np.arange(121, dtype=np.uint16).reshape(11, 11) * 541
    data = np.pad(samples.astype(dtype).view('u1'), ((0, 0), (0, 1)), constant_values=255)
    msg = SimpleNamespace(encoding='mono16', width=11, height=11, step=23,
                          data=data.ravel(), is_bigendian=dtype.startswith('>'))
    np.testing.assert_array_equal(gray(msg), samples / 257.)


@pytest.mark.parametrize('angle,encoding,codec', [(2., 'mono16', None), (-2., 'mono16', None), (2., 'rgb8', None), (-2., 'bgr8', None), (0., 'mono8', None),
                                                  (2., 'rgb8', 'PNG'), (2., 'rgb8', 'JPEG'),
                                                  (-2., 'mono8', 'PNG'), (0., 'mono8', 'JPEG')])
def test_image_bag_to_rotation(tmp_path, angle, encoding, codec):
    w, h = 320, 240
    k = np.array([[230., 0, w / 2], [0, 230., h / 2], [0, 0, 1]])
    mount = Rotation.from_quat(BASE_FROM_CAMERA).as_matrix()
    rotation = Rotation.from_euler('z', angle, degrees=True).as_matrix()
    y, x = np.indices((h, w))
    rays = np.linalg.solve(k, np.stack([x, y, np.ones_like(x)]).reshape(3, -1))
    uv = k @ mount.T @ rotation @ mount @ rays
    uv = (uv[:2] / uv[2]).reshape(2, h, w)
    noise = gaussian_filter(np.random.default_rng(38).normal(size=(h, w)), 1)
    first = np.clip(128 + 135 * noise, 0, 255) if angle else np.full((h, w), 128.)
    second = map_coordinates(first, uv[::-1], order=1, mode='nearest')
    store = get_typestore(Stores.ROS2_HUMBLE)
    types = store.types
    bag = tmp_path / 'bag'
    originals = []
    with Writer(bag, version=8) as writer:
        kind = 'CompressedImage' if codec else 'Image'
        conn = writer.add_connection('/rect', 'sensor_msgs/msg/' + kind, typestore=store)
        for i, frame in enumerate((first, second)):
            header = types['std_msgs/msg/Header'](types['builtin_interfaces/msg/Time'](8, i * 50000000),
                                                  'optical')
            raster = frame.astype(np.uint8)
            if encoding in ('rgb8', 'bgr8'):
                raster = np.stack((raster, raster // 2, raster), axis=-1)
            if codec:
                originals.append(features(raster.astype(float) if raster.ndim == 2
                                          else raster @ [.299, .587, .114]))
                buf = BytesIO()
                Image.fromarray(raster).save(buf, format=codec, quality=90)
                msg = types[conn.msgtype](header, f'{encoding}; {codec.lower()} compressed',
                                          np.frombuffer(buf.getvalue(), dtype=np.uint8))
            else:
                big_endian = encoding == 'mono16' and angle < 0
                if encoding == 'mono16':
                    # Use different high and low bytes to exercise byte order.
                    raster = (raster.astype('u2') * 256).astype('>u2' if big_endian else '<u2').view('u1')
                padded = np.pad(raster.reshape(h, -1), ((0, 0), (0, 3)))
                msg = types[conn.msgtype](header, h, w, encoding, int(big_endian), padded.shape[1], padded.ravel())
            writer.write(conn, 10000000000 + i * 50000000, store.serialize_cdr(msg, conn.msgtype))
    calibration = tmp_path / 'camera.json'
    calibration.write_text(json.dumps(dict(K=k.tolist(), base_from_camera_xyzw=BASE_FROM_CAMERA,
                                           size=[w, h], frame_id='optical')))
    output = tmp_path / 'pairs.jsonl'
    command = [sys.executable, '-m', 'mgts_nav.image_pairs', str(bag), str(output),
               '--topic', '/rect', '--calibration', str(calibration), '--header-offset-s', '.25']
    subprocess.run(command, check=True)
    pair = json.loads(output.read_text())
    if codec == 'PNG':
        assert pair['tracks'] == tracks(*originals)
    assert (pair['t0'], pair['t1']) == (8.25, 8.30)
    assert pair['camera_motion_bound_m'] is None and pair['min_scene_range_m'] is None
    if angle:
        assert estimate(pair)['reason'] == 'missing_motion_or_range_bound'
    output.unlink()
    subprocess.run([*command, '--camera-motion-bound-m', '0', '--min-scene-range-m', '2'], check=True)
    result = estimate(json.loads(output.read_text()))
    if angle:
        assert result['accepted']
        degrees = np.degrees(result['yaw_rad'])
        print(f'{angle:+.1f} -> {degrees:+.4f} deg, {result["inliers"]} inliers')
        assert abs(degrees - angle) < .15
    else:
        assert result['reason'] == 'too_few_tracks' and result['tracks'] == 0


@pytest.mark.parametrize('fmt,reason', [('rgb8; jpeg compressed', 'payload'),
                                        ('16UC1; compressedDepth png', 'format')])
def test_invalid_compressed_payload(fmt, reason):
    with pytest.raises(ValueError, match=reason):
        compressed_gray(SimpleNamespace(format=fmt, data=np.arange(64, dtype=np.uint8)), [320, 240])


@pytest.mark.parametrize('change', ['', 'projection', 'roi', 'rotation', 'frame'])
def test_camera_info_to_image_pairs(tmp_path, change):
    from mgts_nav.image_pairs import convert

    store = get_typestore(Stores.ROS2_HUMBLE)
    types = store.types
    path = tmp_path / 'bag'
    shape = (24, 32)
    rect = Rotation.from_euler('y', 12, degrees=True)
    projection = np.array([[29., 0, 15, 0], [0, 30, 11, 0], [0, 0, 1, 0]])
    with Writer(path, version=8) as writer:
        info = writer.add_connection('/info', 'sensor_msgs/msg/CameraInfo', typestore=store)
        image = writer.add_connection('/rect', 'sensor_msgs/msg/Image', typestore=store)
        for i in range(2):
            header = types['std_msgs/msg/Header'](types['builtin_interfaces/msg/Time'](1, i*50000000),
                                                  'different' if i and change == 'frame' else 'optical')
            roi = types['sensor_msgs/msg/RegionOfInterest'](0, 0, 0, 0, False)
            p = projection.copy()
            r = rect.as_matrix()
            if i and change == 'projection':
                p[0, 0] += 1
            if i and change == 'roi':
                roi.x_offset = 2
            if i and change == 'rotation':
                r[0] *= -1
            msg = types[info.msgtype](header, *shape, 'plumb_bob', np.zeros(5), np.eye(3).ravel(),
                                      r.ravel(), p.ravel(), 0, 0, roi)
            writer.write(info, i, store.serialize_cdr(msg, info.msgtype))
            pixels = np.zeros(shape, dtype=np.uint8)
            msg = types[image.msgtype](header, *shape, 'mono8', 0, shape[1], pixels.ravel())
            writer.write(image, i, store.serialize_cdr(msg, image.msgtype))
    out = tmp_path / 'camera.json'
    cmd = [sys.executable, '-m', 'mgts_nav.camera_info', str(path), str(out), '--topic', '/info',
           '--base-from-camera-xyzw', *map(str, BASE_FROM_CAMERA)]
    run = subprocess.run(cmd, capture_output=True, text=True)
    if change:
        assert run.returncode == 2 and not out.exists(), run.stderr
        return
    assert run.returncode == 0, run.stderr
    profile = json.loads(out.read_text())
    np.testing.assert_array_equal(profile['K'], projection[:, :3])
    ray = np.array([.1, -.2, 1.])
    np.testing.assert_allclose(Rotation.from_quat(profile['base_from_camera_xyzw']).apply(rect.apply(ray)),
                               Rotation.from_quat(BASE_FROM_CAMERA).apply(ray))
    assert convert(path, '/rect', profile, 0, tmp_path/'pairs.jsonl') == 1
    original = out.read_bytes()
    assert subprocess.run(cmd, capture_output=True).returncode == 2
    assert out.read_bytes() == original
