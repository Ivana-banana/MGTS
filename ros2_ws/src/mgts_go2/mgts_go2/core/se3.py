"""Rigid transforms as 4x4 numpy matrices, quaternions in ROS order (x, y, z, w)."""
import numpy as np


def quat_to_rot(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        raise ValueError('zero quaternion')
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def rot_to_quat(r):
    """Return (x, y, z, w) with w >= 0."""
    m = np.asarray(r, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def make(rot=None, trans=None):
    t = np.eye(4)
    if rot is not None:
        t[:3, :3] = rot
    if trans is not None:
        t[:3, 3] = trans
    return t


def from_pose(position, quat):
    return make(quat_to_rot(quat), position)


def inv(t):
    r = t[:3, :3]
    out = np.eye(4)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ t[:3, 3]
    return out


def apply(t, pts):
    return pts @ t[:3, :3].T + t[:3, 3]


def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def so3_exp(w):
    th = np.linalg.norm(w)
    k = hat(w)
    if th < 1e-9:
        return np.eye(3) + k
    return np.eye(3) + np.sin(th) / th * k + (1 - np.cos(th)) / th ** 2 * (k @ k)


def so3_log(r):
    c = np.clip((np.trace(r) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / 2
    if np.pi - th < 1e-6:
        # Rotation by pi: axis from the symmetric part.
        a = np.sqrt(np.maximum((np.diag(r) + 1) / 2, 0))
        i = int(np.argmax(a))
        axis = r[:, i] + np.eye(3)[:, i]
        return axis / np.linalg.norm(axis) * th
    return th / (2 * np.sin(th)) * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])


def yaw_of(t):
    return float(np.arctan2(t[1, 0], t[0, 0]))


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rpy_to_rot(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return rot_z(yaw) @ ry @ rx


def interpolate(t0, t1, a):
    """Pose between t0 (a=0) and t1 (a=1): linear translation, geodesic rotation."""
    r0 = t0[:3, :3]
    dr = so3_exp(a * so3_log(r0.T @ t1[:3, :3]))
    return make(r0 @ dr, (1 - a) * t0[:3, 3] + a * t1[:3, 3])


def delta_norm(t):
    """Translation (m) and rotation angle (rad) of a relative transform."""
    return float(np.linalg.norm(t[:3, 3])), float(np.linalg.norm(so3_log(t[:3, :3])))


def planar(t):
    """Keep only x, y and yaw of a transform (roll = pitch = z = 0)."""
    return make(rot_z(yaw_of(t)), (t[0, 3], t[1, 3], 0.0))


def planar_prior_mask():
    """Body-frame dims (rx, ry, rz, tx, ty, tz) that a planar correction may not change."""
    return (True, True, False, False, False, True)
