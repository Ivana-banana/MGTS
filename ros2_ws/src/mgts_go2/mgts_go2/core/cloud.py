"""PointCloud2 decoding and point-set helpers shared by the ROS node and offline tools.

Works with any message object exposing ``fields`` (name, offset, datatype),
``point_step``, ``width``, ``height``, ``is_bigendian`` and ``data``: both rclpy
messages and ``rosbags`` deserialized messages qualify.
"""
import numpy as np

# sensor_msgs/PointField datatypes
_TYPES = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}


def decode(msg, wanted=('x', 'y', 'z', 'intensity', 'time')):
    """Return a dict name -> 1-D float64 array for the wanted fields that exist."""
    endian = '>' if msg.is_bigendian else '<'
    present = {f.name: f for f in msg.fields}
    names, formats, offsets = [], [], []
    for name in wanted:
        f = present.get(name)
        if f is None:
            continue
        names.append(name)
        formats.append(endian + _TYPES[int(f.datatype)])
        offsets.append(int(f.offset))
    if not {'x', 'y', 'z'} <= set(names):
        raise ValueError('PointCloud2 without x/y/z fields: %s' % sorted(present))
    dt = np.dtype({'names': names, 'formats': formats, 'offsets': offsets, 'itemsize': int(msg.point_step)})
    n = int(msg.width) * int(msg.height)
    raw = np.frombuffer(bytes(msg.data) if not isinstance(msg.data, np.ndarray) else msg.data.tobytes(), dtype=dt, count=n)
    return {k: raw[k].astype(np.float64) for k in names}


def xyz(fields):
    return np.column_stack([fields['x'], fields['y'], fields['z']])


def valid_mask(pts, min_range=0.0, max_range=np.inf):
    """Finite, not the all-zero padding Unitree adds, and within range limits."""
    finite = np.all(np.isfinite(pts), axis=1)
    nonzero = np.any(pts != 0.0, axis=1)
    r = np.linalg.norm(pts, axis=1)
    return finite & nonzero & (r >= min_range) & (r <= max_range)


def voxel_downsample(pts, size, extra=None):
    """Keep the centroid of each occupied voxel. ``extra`` columns are averaged too."""
    if len(pts) == 0:
        return pts if extra is None else (pts, extra)
    keys = np.floor(pts / size).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    m = len(counts)
    out = np.zeros((m, 3))
    np.add.at(out, inverse, pts)
    out /= counts[:, None]
    if extra is None:
        return out
    ex = np.zeros((m,) + extra.shape[1:])
    np.add.at(ex, inverse, extra)
    ex /= counts.reshape((-1,) + (1,) * (extra.ndim - 1))
    return out, ex


def voxel_keys(pts, size):
    return np.floor(pts / size).astype(np.int64)


def pack_keys(keys):
    """Pack integer voxel coordinates (|k| < 2**20) into one int64 per voxel."""
    k = keys + (1 << 20)
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def unpack_keys(packed):
    mask = (1 << 21) - 1
    return np.column_stack([(packed >> 42) & mask, (packed >> 21) & mask, packed & mask]) - (1 << 20)
