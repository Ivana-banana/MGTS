"""Extract visual_fusion JSONL from consecutive rectified sensor_msgs/Image frames."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial.distance import cdist

from .bag import reader, stamp


def gray(msg):
    channels = {'mono8': 1, 'rgb8': 3, 'bgr8': 3}.get(msg.encoding)
    if channels is None:
        raise ValueError(f'Unsupported image encoding {msg.encoding!r}')
    w, h = msg.width, msg.height
    if min(w, h) < 11 or msg.step < w * channels or len(msg.data) != h * msg.step:
        raise ValueError('Invalid image dimensions, stride or data length')
    pixels = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, msg.step)[:, :w * channels]
    if channels == 1:
        return pixels.astype(float)
    rgb = pixels.reshape(h, w, 3)
    return (rgb if msg.encoding == 'rgb8' else rgb[:, :, ::-1]) @ [0.299, 0.587, 0.114]


def features(image):
    dy, dx = np.gradient(gaussian_filter(image, 1))
    xx, xy, yy = [gaussian_filter(v, 1.5) for v in (dx * dx, dx * dy, dy * dy)]
    score = (xx + yy - np.hypot(xx - yy, 2 * xy)) / 2
    peaks = (score == maximum_filter(score, 9)) & (score > max(1., score.max() * .01))
    y, x = np.nonzero(peaks[5:-5, 5:-5])
    y, x = y + 5, x + 5
    order = np.argsort(-score[y, x], kind='stable')[:400]
    y, x = y[order], x[order]
    patches = np.lib.stride_tricks.sliding_window_view(image, (11, 11))[y - 5, x - 5]
    patches = patches.reshape(len(x), 121)
    patches -= patches.mean(axis=1, keepdims=True)
    patches /= np.maximum(np.linalg.norm(patches, axis=1, keepdims=True), 1e-12)
    return np.column_stack((x, y)), patches


def tracks(first, second):
    p, a = first
    q, b = second
    if min(len(p), len(q)) < 2:
        return []
    distances = cdist(a, b)
    matches = distances.argmin(axis=1)
    best, runner_up = np.partition(distances, 1, axis=1)[:, :2].T
    keep = (distances.argmin(axis=0)[matches] == np.arange(len(p))) & (best < .75 * runner_up)
    return [[int(i), *p[i].tolist(), *q[matches[i]].tolist()] for i in np.flatnonzero(keep)]


def convert(bag_path, topic, calibration, offset, output, motion=None, distance=None):
    k = np.asarray(calibration['K'], dtype=float)
    quat = np.asarray(calibration['base_from_camera_xyzw'], dtype=float)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0, 0, 1]) or not np.isclose(k[1, 0], 0)
            or quat.shape != (4,) or not np.isfinite(quat).all()
            or abs(np.linalg.norm(quat) - 1) > .001):
        raise ValueError('Invalid rectified intrinsics or mount quaternion')
    if (not np.isfinite(offset) or motion is not None and (not np.isfinite(motion) or motion < 0)
            or distance is not None and (not np.isfinite(distance) or distance <= 0)):
        raise ValueError('Invalid time offset, motion bound or minimum range')
    with reader(bag_path) as bag:
        conns = [c for c in bag.connections if c.topic == topic]
        if not conns or any(c.msgtype != 'sensor_msgs/msg/Image' for c in conns):
            raise ValueError(f'Expected sensor_msgs/Image on {topic}')
        with NamedTemporaryFile(dir=output.parent, delete=False) as tmp:
            staged = Path(tmp.name)
        try:
            count = 0
            previous = None
            with staged.open('w', encoding='utf-8') as stream:
                for conn, _, raw in bag.messages(connections=conns):
                    msg = bag.deserialize(raw, conn.msgtype)
                    if (msg.header.frame_id != calibration['frame_id']
                            or [msg.width, msg.height] != calibration['size']):
                        raise ValueError('Image frame or resolution differs from calibration')
                    t = stamp(msg) + offset
                    current = features(gray(msg))
                    if previous is not None:
                        if t <= last:
                            raise ValueError('Camera header timestamps must increase')
                        pair = dict(t0=last, t1=t, K=k.tolist(), base_from_camera_xyzw=quat.tolist(),
                                    camera_motion_bound_m=motion, min_scene_range_m=distance,
                                    tracks=tracks(previous, current))
                        stream.write(json.dumps(pair, allow_nan=False) + '\n')
                        count += 1
                    previous, last = current, t
            if not count:
                raise ValueError('Need at least two images')
            # On the supported Windows host rename refuses to overwrite an existing output.
            if output.exists():
                raise FileExistsError(output)
            staged.rename(output)
            return count
        finally:
            staged.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--topic', required=True)
    parser.add_argument('--calibration', required=True, type=Path)
    parser.add_argument('--header-offset-s', required=True, type=float)
    parser.add_argument('--camera-motion-bound-m', type=float)
    parser.add_argument('--min-scene-range-m', type=float)
    args = parser.parse_args()
    try:
        print(convert(args.bag, args.topic, json.loads(args.calibration.read_text(encoding='utf-8-sig')),
                      args.header_offset_s, args.output, args.camera_motion_bound_m, args.min_scene_range_m))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'{exc}\n')
