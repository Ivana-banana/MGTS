"""Extract a full-resolution rectified image_pairs calibration from ROS CameraInfo."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .bag import reader


def calibration(msg, mount):
    # ROS 1 uses uppercase matrix field names and ROS 2 uses lowercase.
    k, r, p = [np.asarray(getattr(msg, name, getattr(msg, name.upper(), None))).reshape(shape)
               for name, shape in [('k', (3, 3)), ('r', (3, 3)), ('p', (3, 4))]]
    if not msg.header.frame_id.strip() or min(msg.width, msg.height) < 11:
        raise ValueError('Invalid optical frame or image dimensions')
    roi = msg.roi
    if (max(msg.binning_x, msg.binning_y) > 1 or roi.x_offset or roi.y_offset
            or (roi.width, roi.height) not in [(0, 0), (msg.width, msg.height)] or roi.do_rectify):
        raise ValueError('Expected full-resolution CameraInfo without ROI rectification or binning')
    if (not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.isfinite(p).all()
            or min(p[0, 0], p[1, 1]) <= 0 or not np.allclose(p[2], [0, 0, 1, 0])
            or not np.isclose(p[1, 0], 0)):
        raise ValueError('Uncalibrated camera or invalid projection matrix')
    if (not np.allclose(r.T @ r, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(r), 1., atol=1e-6, rtol=0)):
        raise ValueError('Invalid CameraInfo rectification rotation')
    orientation = mount * Rotation.from_matrix(r.T)
    return dict(size=[msg.width, msg.height], frame_id=msg.header.frame_id, K=p[:, :3].tolist(),
                base_from_camera_xyzw=orientation.as_quat().tolist())


def extract(path, topic, quaternion):
    if not np.isfinite(quaternion).all() or abs(np.linalg.norm(quaternion) - 1) > .001:
        raise ValueError('Expected a unit base-from-camera quaternion')
    mount = Rotation.from_quat(quaternion)
    profile = None
    with reader(path) as bag:
        conns = [c for c in bag.connections if c.topic == topic]
        if not conns or any(c.msgtype != 'sensor_msgs/msg/CameraInfo' for c in conns):
            raise ValueError(f'Expected CameraInfo on {topic}')
        for conn, timestamp, raw in bag.messages(connections=conns):
            candidate = calibration(bag.deserialize(raw, conn.msgtype), mount)
            if profile is not None and candidate != profile:
                raise ValueError(f'Rectified camera calibration changed on {topic} at {timestamp} ns')
            profile = candidate
    if profile is None:
        raise ValueError(f'Empty CameraInfo topic {topic}')
    return profile


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--topic', required=True)
    parser.add_argument('--base-from-camera-xyzw', nargs=4, type=float, required=True)
    args = parser.parse_args()
    try:
        profile = extract(args.bag, args.topic, args.base_from_camera_xyzw)
        payload = json.dumps(profile, indent=2, allow_nan=False)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(payload + '\n')
    except (ValueError, OSError) as exc:
        parser.exit(2, f'{exc}\n')
