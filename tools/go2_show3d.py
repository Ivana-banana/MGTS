"""Render the official Go2 inspection with the Go2 model, published 3D poses and measured joints.

The base pose comes from /mgts/pose3d of the result bag. The twelve joint angles come from
/lowstate of the inspection recording. Walls are boxes built from observed cells of the
mapping map. An optional model detour (#212) is shown afterwards under an explicit caption.
"""
import argparse
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

# Unitree LowState motor order is FR, FL, RR, RL (hip, thigh, calf each).
# The Menagerie MJCF orders legs FL, FR, RL, RR. Checked on the official inspection:
# with this order a longer left stance leg correlates with positive IMU roll.
MOTOR_TO_MODEL = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
LEGS = ('FL', 'FR', 'RL', 'RR')
VOXEL_M = 0.12
TRAIL_STEP_M = 0.05
RED = (1.0, 0.0, 0.196)  # MTS red #FF0032
DETOUR_RGB = (0.0, 0.47, 0.84)


def model_joints(motor_q):
    """Reorder LowState motor angles into the MJCF joint order."""
    return np.asarray(motor_q)[..., list(MOTOR_TO_MODEL)]


def header_clock(delivery, ref_delivery, ref_header):
    """Express delivery times on the header clock through a reference topic with both times."""
    offset = np.asarray(ref_delivery) - np.asarray(ref_header)
    return np.asarray(delivery) - np.interp(delivery, ref_delivery, offset)


def sample_poses(times, pose_t, pos, quat_xyzw):
    """Interpolate position linearly and orientation by slerp; clamp outside the pose range."""
    t = np.clip(times, pose_t[0], pose_t[-1])
    xyz = np.column_stack([np.interp(t, pose_t, pos[:, i]) for i in range(3)])
    rot = Slerp(pose_t, Rotation.from_quat(quat_xyzw))(t)
    idx = np.searchsorted(pose_t, t)
    nearest = np.minimum(np.abs(pose_t[np.clip(idx, 0, len(pose_t) - 1)] - t),
                         np.abs(pose_t[np.clip(idx - 1, 0, len(pose_t) - 1)] - t))
    return xyz, rot, nearest


def nearest_index(sorted_t, times):
    idx = np.clip(np.searchsorted(sorted_t, times), 1, len(sorted_t) - 1)
    left = sorted_t[idx - 1]
    return np.where(np.abs(times - left) <= np.abs(sorted_t[idx] - times), idx - 1, idx)


def wall_boxes(points, floor_z, cut_z, voxel=VOXEL_M, min_cells=3, floor_snap_cells=2):
    """Observed upright runs of cells as boxes (x0, x1, y0, y1, z0, z1) in metres.

    A run needs min_cells stacked cells and a neighbour at the same height. Runs whose bottom
    is within floor_snap_cells of the floor are extended down to it. Horizontal gaps stay open.
    """
    cells = np.unique(np.floor(np.asarray(points) / voxel).astype(int), axis=0)
    columns = defaultdict(set)
    for x, y, z in cells:
        if (z + 0.5) * voxel > floor_z + voxel:
            columns[x, y].add(int(z))
    runs = {}
    for (x, y), levels in columns.items():
        for bottom in levels:
            if bottom - 1 in levels:
                continue
            top = bottom
            while top + 1 in levels:
                top += 1
            if top - bottom + 1 >= min_cells:
                runs.setdefault((x, y), []).append((bottom, top))
    occupied = {(x, y, z) for (x, y), spans in runs.items() for b, t in spans for z in range(b, t + 1)}
    neighbours = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    spans = []
    for (x, y), column in runs.items():
        for bottom, top in column:
            if any((x + dx, y + dy, z) in occupied for dx, dy in neighbours for z in range(bottom, top + 1)):
                z0 = bottom * voxel
                if z0 - floor_z <= floor_snap_cells * voxel:
                    z0 = floor_z
                z1 = min((top + 1) * voxel, cut_z)
                if z1 - z0 > 0.02:
                    spans.append((y, round(z0, 6), round(z1, 6), x))
    boxes = []
    spans.sort()
    for y, z0, z1, x in spans:
        last = boxes[-1] if boxes else None
        if last and last[2] == y and last[4] == z0 and last[5] == z1 and last[1] == x:
            last[1] = x + 1
        else:
            boxes.append([x, x + 1, y, y, z0, z1])
    return [(x0 * voxel, x1 * voxel, y * voxel, (y + 1) * voxel, z0, z1) for x0, x1, y, _, z0, z1 in boxes]


def smooth_heading(yaw, sigma):
    return gaussian_filter1d(np.unwrap(yaw), sigma, mode='nearest')


def resample_polyline(xy, step):
    xy = np.asarray(xy, float)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    grid = np.arange(0.0, s[-1] + 1e-9, step)
    return np.column_stack([np.interp(grid, s, xy[:, 0]), np.interp(grid, s, xy[:, 1])]), s[-1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ru(value, digits=1):
    return f'{value:.{digits}f}'.replace('.', ',')


# ---------------------------------------------------------------- data


def load_result_poses(result_dir, topic='/mgts/pose3d'):
    from mgts_nav.bag import reader, stamp
    rows = []
    with reader(Path(result_dir) / 'bag') as bag:
        sel = [c for c in bag.connections if c.topic == topic]
        if not sel:
            raise ValueError(f'{result_dir}: no {topic} in result bag')
        for c, _, raw in bag.messages(connections=sel):
            m = bag.deserialize(raw, c.msgtype)
            p, q = m.pose.position, m.pose.orientation
            rows.append([stamp(m), p.x, p.y, p.z, q.x, q.y, q.z, q.w])
    rows = np.array(sorted(rows))
    if np.any(np.diff(rows[:, 0]) <= 0):
        raise ValueError('result poses must have strictly increasing stamps')
    return rows[:, 0], rows[:, 1:4], rows[:, 4:8]


def load_lowstate(recording):
    """Joint angles and foot loads on the header clock of the recording."""
    from mgts_nav.bag import reader, stamp
    low, odom = [], []
    with reader(recording) as bag:
        sel = [c for c in bag.connections if c.topic in ('/lowstate', '/utlidar/robot_odom')]
        for c, ns, raw in bag.messages(connections=sel):
            m = bag.deserialize(raw, c.msgtype)
            if c.topic == '/lowstate':
                low.append([ns * 1e-9, *[m.motor_state[i].q for i in range(12)], *m.foot_force])
            else:
                odom.append([ns * 1e-9, stamp(m)])
    low, odom = np.array(low), np.array(odom)
    if not len(low) or not len(odom):
        raise ValueError(f'{recording}: /lowstate and /utlidar/robot_odom are required')
    offset = odom[:, 0] - odom[:, 1]
    t = header_clock(low[:, 0], odom[:, 0], odom[:, 1])
    order = np.argsort(t, kind='stable')
    clock = dict(reference='/utlidar/robot_odom', pairs=len(odom),
                 delivery_minus_header_s=dict(zip(('min', 'p50', 'max'),
                                                  np.percentile(offset, [0, 50, 100]).round(6).tolist())))
    return t[order], model_joints(low[order, 1:13]), low[order, 13:17], clock


# ---------------------------------------------------------------- scene


def build_model(model_xml, boxes, floor_z, bounds, size, detour=None):
    import mujoco
    spec = mujoco.MjSpec.from_file(str(model_xml))
    w, h = size
    spec.visual.global_.offwidth, spec.visual.global_.offheight = w, h
    spec.visual.quality.shadowsize = 8192
    spec.visual.quality.offsamples = 8
    spec.visual.headlight.ambient = [0.30, 0.30, 0.32]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    spec.visual.headlight.specular = [0.05, 0.05, 0.05]
    spec.visual.rgba.haze = [0.93, 0.94, 0.96, 1]
    spec.visual.map.zfar = 80
    spec.add_texture(name='sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.97, 0.97, 0.98], rgb2=[0.86, 0.88, 0.91], width=512, height=3072)
    spec.add_texture(name='grid', type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     mark=mujoco.mjtMark.mjMARK_EDGE, rgb1=[0.74, 0.75, 0.77], rgb2=[0.74, 0.75, 0.77],
                     markrgb=[0.64, 0.65, 0.68], width=512, height=512)
    floor = spec.add_material(name='floor_grid', texrepeat=[0.5, 0.5], texuniform=True, reflectance=0.06)
    floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'grid'
    spec.add_material(name='wall', rgba=[0.80, 0.83, 0.87, 1], specular=0.1, shininess=0.1)
    spec.add_material(name='box', rgba=[0.95, 0.62, 0.10, 1], specular=0.2)
    (x0, x1), (y0, y1) = bounds
    static = dict(contype=0, conaffinity=0, group=0)
    spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, material='floor_grid',
                            size=[(x1 - x0) / 2 + 6, (y1 - y0) / 2 + 6, 0.05],
                            pos=[(x0 + x1) / 2, (y0 + y1) / 2, floor_z], **static)
    for i, (bx0, bx1, by0, by1, bz0, bz1) in enumerate(boxes):
        spec.worldbody.add_geom(name=f'wall{i}', type=mujoco.mjtGeom.mjGEOM_BOX, material='wall',
                                size=[(bx1 - bx0) / 2, (by1 - by0) / 2, (bz1 - bz0) / 2],
                                pos=[(bx0 + bx1) / 2, (by0 + by1) / 2, (bz0 + bz1) / 2], **static)
    if detour is not None:
        cx, cy = detour['obstacle']['center_map3d']
        sx, sy = detour['obstacle']['size_m']
        yaw = detour['obstacle']['yaw_in_map3d_rad']
        quat = Rotation.from_euler('z', yaw).as_quat()[[3, 0, 1, 2]]
        spec.worldbody.add_geom(name='model_box', type=mujoco.mjtGeom.mjGEOM_BOX, material='box',
                                size=[sx / 2, sy / 2, 0.25], pos=[cx, cy, floor_z + 0.25], quat=quat, **static)
    spec.worldbody.add_light(name='sun', pos=[(x0 + x1) / 2, (y0 + y1) / 2, 30], dir=[0.25, 0.35, -1],
                             diffuse=[0.35, 0.35, 0.35], specular=[0.1, 0.1, 0.1], castshadow=False,
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.body('base').add_light(name='key', mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM, pos=[0.8, -0.6, 3.2],
                                dir=[-0.25, 0.19, -1], diffuse=[0.32, 0.32, 0.32], specular=[0.1, 0.1, 0.1],
                                cutoff=50, exponent=2, castshadow=True, type=mujoco.mjtLightType.mjLIGHT_SPOT)
    for geom in spec.geoms:
        geom.contype, geom.conaffinity = 0, 0
    return spec.compile()


def set_robot(data, pos, rot, joints):
    data.qpos[0:3] = pos
    data.qpos[3:7] = rot.as_quat()[[3, 0, 1, 2]]
    data.qpos[7:19] = joints


def foot_heights(model, data, pos_rot_joints):
    import mujoco
    feet = [model.geom(name).id for name in LEGS]
    radius = model.geom_size[feet[0], 0]
    lows = []
    for pos, rot, joints in pos_rot_joints:
        set_robot(data, pos, rot, joints)
        mujoco.mj_kinematics(model, data)
        lows.append(data.geom_xpos[feet, 2].min() - radius)
    return np.array(lows)


def add_capsule(scene, a, b, width, rgba):
    import mujoco
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))
    g.emission = 0.35
    scene.ngeom += 1


def add_arrow(scene, a, b, width, rgba):
    import mujoco
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(a, float), np.asarray(b, float))
    g.emission = 0.3
    scene.ngeom += 1


def add_polyline(scene, xy, z, width, rgba, dashed=False):
    for i in range(len(xy) - 1):
        if dashed and i % 2:
            continue
        add_capsule(scene, (*xy[i], z), (*xy[i + 1], z), width, rgba)


# ---------------------------------------------------------------- overlay


class Overlay:
    def __init__(self, size, wall_boxes_xy, route_xy, bounds, font_dir=None):
        self.w, self.h = size
        self.fonts = {}
        regular, bold = find_fonts(font_dir)
        self.regular, self.bold = regular, bold
        self.bounds = bounds
        self.inset = self._inset(wall_boxes_xy, route_xy)

    def font(self, size, bold=False):
        key = size, bold
        if key not in self.fonts:
            self.fonts[key] = ImageFont.truetype(self.bold if bold else self.regular, size)
        return self.fonts[key]

    def _inset(self, boxes, route_xy):
        (x0, x1), (y0, y1) = self.bounds
        # The long axis of the explored area runs along +y; show it left to right.
        self.scale = min(620 / (y1 - y0), 300 / (x1 - x0))
        iw, ih = int((y1 - y0) * self.scale) + 28, int((x1 - x0) * self.scale) + 28
        img = Image.new('RGBA', (iw, ih), (255, 255, 255, 228))
        draw = ImageDraw.Draw(img)
        for bx0, bx1, by0, by1, _, _ in boxes:
            a, b = self.to_inset(bx0, by0), self.to_inset(bx1, by1)
            draw.rectangle([min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])], fill=(118, 128, 140, 255))
        if route_xy is not None:
            draw.line([self.to_inset(x, y) for x, y in route_xy], fill=(200, 200, 205, 255), width=2)
        draw.rectangle([0, 0, iw - 1, ih - 1], outline=(190, 194, 200, 255), width=2)
        return img

    def to_inset(self, x, y):
        (x0, x1), (y0, _) = self.bounds
        return 14 + (y - y0) * self.scale, 14 + (x - x0) * self.scale

    def draw(self, frame, lines, trail_xy, robot_xy, robot_yaw, detour_xy=None, banner=None, legend=()):
        img = Image.fromarray(frame).convert('RGBA')
        layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        pad = 36
        # Title block.
        title, subtitle = lines[0], lines[1:]
        tw = max([d.textlength(title, font=self.font(40, True))] + [d.textlength(s, font=self.font(24)) for s in subtitle])
        d.rounded_rectangle([pad - 18, pad - 14, pad + tw + 18, pad + 52 + 34 * len(subtitle)], 14, fill=(255, 255, 255, 214))
        d.text((pad, pad), title, font=self.font(40, True), fill=(20, 22, 26, 255))
        for i, s in enumerate(subtitle):
            d.text((pad, pad + 56 + 34 * i), s, font=self.font(24), fill=(60, 64, 72, 255))
        # Inset map with trail and robot.
        inset = self.inset.copy()
        di = ImageDraw.Draw(inset)
        if detour_xy is not None and len(detour_xy) > 1:
            di.line([self.to_inset(x, y) for x, y in detour_xy], fill=(0, 120, 214, 255), width=3)
        if trail_xy is not None and len(trail_xy) > 1:
            di.line([self.to_inset(x, y) for x, y in trail_xy], fill=(255, 0, 50, 255), width=3)
        cx, cy = self.to_inset(*robot_xy)
        # Map +x points down and +y right in the inset, so a heading becomes (sin, cos) on screen.
        fx, fy = np.sin(robot_yaw), np.cos(robot_yaw)
        tri = [(cx + 13 * fy, cy + 13 * fx), (cx - 8 * fy + 7 * fx, cy - 8 * fx - 7 * fy),
               (cx - 8 * fy - 7 * fx, cy - 8 * fx + 7 * fy)]
        di.polygon(tri, fill=(20, 22, 26, 255), outline=(255, 255, 255, 255))
        ix, iy = self.w - inset.width - pad, self.h - inset.height - pad
        layer.alpha_composite(inset, (ix, iy))
        d.text((ix, iy - 34), 'Вид сверху: весь пройденный путь', font=self.font(22, True), fill=(40, 44, 50, 255))
        # Legend.
        if legend:
            y = self.h - pad - 34 * len(legend)
            lw = max(d.textlength(text, font=self.font(24)) for _, text in legend) + 74
            d.rounded_rectangle([pad - 18, y - 14, pad + lw, self.h - pad + 8], 14, fill=(255, 255, 255, 214))
            for color, text in legend:
                d.rounded_rectangle([pad, y + 9, pad + 40, y + 17], 4, fill=color)
                d.text((pad + 54, y), text, font=self.font(24), fill=(40, 44, 50, 255))
                y += 34
        if banner:
            bw = d.textlength(banner, font=self.font(34, True)) + 48
            bx = (self.w - bw) / 2
            d.rounded_rectangle([bx, pad - 14, bx + bw, pad + 42], 12, fill=(0, 92, 170, 236))
            d.text((bx + 24, pad - 6), banner, font=self.font(34, True), fill=(255, 255, 255, 255))
        img.alpha_composite(layer)
        return np.asarray(img.convert('RGB'))


def find_fonts(font_dir=None):
    candidates = []
    if font_dir:
        candidates.append((Path(font_dir) / 'NotoSans-Regular.ttf', Path(font_dir) / 'NotoSans-Bold.ttf'))
    candidates.append((Path('/usr/share/fonts/noto/NotoSans-Regular.ttf'), Path('/usr/share/fonts/noto/NotoSans-Bold.ttf')))
    candidates.append((Path('C:/Windows/Fonts/segoeui.ttf'), Path('C:/Windows/Fonts/segoeuib.ttf')))
    import matplotlib
    mpl = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
    candidates.append((mpl / 'DejaVuSans.ttf', mpl / 'DejaVuSans-Bold.ttf'))
    for regular, bold in candidates:
        if regular.is_file() and bold.is_file():
            return str(regular), str(bold)
    raise FileNotFoundError('no Cyrillic font found; pass --font-dir')


# ---------------------------------------------------------------- render


def chase_camera(cam, lookat, yaw, distance, elevation):
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = np.degrees(yaw)
    cam.elevation = elevation


def render(args):
    import mujoco
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    w, h = args.size
    fps = args.fps

    pose_t, pose_xyz, pose_q = load_result_poses(args.result, args.pose_topic)
    low_t, low_q, low_force, clock = load_lowstate(args.inspection)
    points = np.load(Path(args.result) / 'map.npz')['xyz']

    # Frame times on the header clock: real-time 1x from the first to the last published pose.
    t_end = pose_t[-1] if args.seconds is None else min(pose_t[-1], pose_t[0] + args.seconds)
    frame_t = np.arange(pose_t[0], t_end + 1e-9, 1.0 / fps)
    xyz, rot, pose_gap = sample_poses(frame_t, pose_t, pose_xyz, pose_q)
    j_idx = nearest_index(low_t, frame_t)
    joint_dt = np.abs(low_t[j_idx] - frame_t)
    joints = low_q[j_idx]

    # Floor height from forward kinematics: published base pose + measured joints, lowest foot.
    probe = mujoco.MjModel.from_xml_path(str(args.model))
    probe_data = mujoco.MjData(probe)
    k = nearest_index(low_t, pose_t)
    lows = foot_heights(probe, probe_data, zip(pose_xyz, Rotation.from_quat(pose_q), low_q[k]))
    floor_z = float(np.median(lows))
    near = np.min(np.linalg.norm(points[:, None, :2] - pose_xyz[None, ::25, :2], axis=2), axis=1) < 1.5
    map_floor = points[near & (points[:, 2] < floor_z + 0.25), 2]

    pad = 2.5
    bounds = ((pose_xyz[:, 0].min() - pad, pose_xyz[:, 0].max() + pad), (pose_xyz[:, 1].min() - pad, pose_xyz[:, 1].max() + pad))
    inside = ((points[:, 0] > bounds[0][0] - 3) & (points[:, 0] < bounds[0][1] + 3)
              & (points[:, 1] > bounds[1][0] - 3) & (points[:, 1] < bounds[1][1] + 3))
    # Cells the robot body passed through during inspection cannot be static walls; do not draw them.
    from scipy.spatial import cKDTree
    swept = cKDTree(pose_xyz[:, :2]).query(points[:, :2], distance_upper_bound=args.swept_radius)[0] < args.swept_radius
    boxes = wall_boxes(points[inside & ~swept], floor_z, floor_z + args.wall_cut)
    detour = json.loads(Path(args.detour).read_text(encoding='utf-8')) if args.detour else None
    model = build_model(args.model, boxes, floor_z, bounds, (w, h), detour)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, h, w, max_geom=20000)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    opt = mujoco.MjvOption()

    yaw = np.array([r.as_euler('ZYX')[0] for r in rot])
    sm_yaw = smooth_heading(yaw, 0.9 * fps)
    sm_xyz = gaussian_filter1d(xyz, 0.25 * fps, axis=0, mode='nearest')
    vel = np.gradient(gaussian_filter1d(xyz[:, :2], 0.4 * fps, axis=0, mode='nearest'), axis=0) * fps
    speed = np.linalg.norm(vel, axis=1)

    overlay = Overlay((w, h), boxes, None, bounds, args.font_dir)
    legend = [((255, 0, 50, 255), 'пройденный путь по опубликованным позам'),
              ((255, 0, 50, 140), 'направление движения'),
              ((170, 178, 190, 255), f'стены по карте обхода, срез {ru(args.wall_cut, 1)} м над полом')]
    title = ['Go2 · официальная инспекция, воспроизведение 1×',
             'позы /mgts/pose3d прогона a5ad606 · углы 12 суставов /lowstate',
             'модель Unitree Go2 из MuJoCo Menagerie']
    video = out / args.name
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                           '-s', f'{w}x{h}', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'medium',
                           '-crf', str(args.crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)],
                          stdin=subprocess.PIPE)
    trail = [xyz[0, :2].copy()]
    trail_idx = [0]
    stills = {}
    still_frames = {int(round(s * fps)) for s in args.still}
    detour_info = None
    try:
        intro = int(args.intro * fps)
        for i in range(intro):
            # Opening: slow pull-in from an overview to the chase view at the first pose.
            a = 0.5 - 0.5 * np.cos(np.pi * i / max(intro - 1, 1))
            set_robot(data, xyz[0], rot[0], joints[0])
            mujoco.mj_forward(model, data)
            center = np.array([np.mean(bounds[0]), np.mean(bounds[1]), floor_z])
            look = (1 - a) * center + a * (sm_xyz[0] + [0, 0, 0.05])
            chase_camera(cam, look, sm_yaw[0] - (1 - a) * 0.6, 22 * (1 - a) + args.distance * a, -62 * (1 - a) + args.elevation * a)
            renderer.update_scene(data, cam, opt)
            frame = renderer.render()
            ff.stdin.write(overlay.draw(frame, title, None, xyz[0, :2], yaw[0], legend=legend).tobytes())
        for i in range(len(frame_t)):
            set_robot(data, xyz[i], rot[i], joints[i])
            mujoco.mj_forward(model, data)
            chase_camera(cam, sm_xyz[i] + [0, 0, 0.05], sm_yaw[i], args.distance, args.elevation)
            if np.linalg.norm(xyz[i, :2] - trail[-1]) >= TRAIL_STEP_M:
                trail.append(xyz[i, :2].copy())
                trail_idx.append(i)
            renderer.update_scene(data, cam, opt)
            path = np.array(trail + [xyz[i, :2]])
            add_polyline(renderer.scene, path, floor_z + 0.012, 0.028, (*RED, 1.0))
            if speed[i] > 0.08:
                head = np.arctan2(vel[i, 1], vel[i, 0])
                d = np.array([np.cos(head), np.sin(head)])
                a0 = np.r_[xyz[i, :2] + 0.42 * d, floor_z + 0.02]
                add_arrow(renderer.scene, a0, a0 + np.r_[0.45 * d, 0], 0.05, (*RED, min(1.0, 0.4 + speed[i])))
            frame = renderer.render()
            elapsed = frame_t[i] - pose_t[0]
            lines = title[:1] + [f't = {ru(elapsed)} с из {ru(pose_t[-1] - pose_t[0])} с · скорость {ru(speed[i], 2)} м/с',
                                 title[1], title[2]]
            frame = overlay.draw(frame, lines, path, xyz[i, :2], yaw[i], legend=legend)
            if i in still_frames:
                stills[i] = frame
            ff.stdin.write(frame.tobytes())
        if detour is not None and args.seconds is None:
            detour_info = render_detour(args, detour, model, data, renderer, cam, opt, overlay, ff, xyz, low_t, low_q,
                                        pose_t, pose_xyz, path, floor_z, fps)
        for _ in range(int(args.outro * fps)):
            ff.stdin.write(frame.tobytes())
    finally:
        ff.stdin.close()
        code = ff.wait()
    if code:
        raise RuntimeError(f'ffmpeg exited with {code}')
    still_paths = []
    for i, frame in sorted(stills.items()):
        p = out / f'still-{ru(frame_t[i] - pose_t[0])}s.png'.replace(',', '_')
        Image.fromarray(frame).save(p)
        still_paths.append(p.name)

    sources = {
        'result_bag': str(Path(args.result) / 'bag' / 'result_0.mcap'),
        'map': str(Path(args.result) / 'map.npz'),
        'inspection': str(next(Path(args.inspection).glob('*.mcap'))),
        'model': str(args.model),
    }
    manifest = dict(
        video=video.name, size=[w, h], fps=fps, frames_inspection=len(frame_t),
        seconds_inspection=round(float(frame_t[-1] - frame_t[0]), 3), intro_s=args.intro, outro_s=args.outro,
        poses=dict(topic=args.pose_topic, count=len(pose_t), first_header_s=float(pose_t[0]), last_header_s=float(pose_t[-1]),
                   max_gap_s=round(float(np.diff(pose_t).max()), 4),
                   frame_to_nearest_pose_s_p50_p99_max=np.percentile(pose_gap, [50, 99, 100]).round(4).tolist(),
                   between_poses='position linear, orientation slerp'),
        joints=dict(topic='/lowstate', samples=len(low_t), motor_order='FR,FL,RR,RL -> model FL,FR,RL,RR',
                    clock=clock, frame_to_sample_s_p50_p99_max=np.percentile(joint_dt, [50, 99, 100]).round(5).tolist()),
        floor=dict(z_m=round(floor_z, 4), method='median lowest foot over published poses with measured joints',
                   lowest_foot_minus_floor_m_p5_p50_p95=np.percentile(lows - floor_z, [5, 50, 95]).round(4).tolist(),
                   map_points_near_route_below_floor_plus_0_25_m=dict(count=int(len(map_floor)),
                       z_p10_p50_p90=np.percentile(map_floor, [10, 50, 90]).round(4).tolist() if len(map_floor) else None)),
        walls=dict(boxes=len(boxes), voxel_m=VOXEL_M, cut_above_floor_m=args.wall_cut, source='map.npz of run a5ad606',
                   hidden_points_within_swept_radius=int(swept.sum()), swept_radius_m=args.swept_radius),
        model_detour=detour_info, stills=still_paths,
        sha256={k: sha256(v) for k, v in sources.items() if Path(v).is_file()},
        sources=sources,
    )
    manifest['sha256']['video'] = sha256(video)
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def gait_clip(low_t, low_q, pose_t, pose_xyz, seconds=2.0):
    """Longest-straight measured walking window: joint angles to loop and its mean speed."""
    best = None
    for start in np.arange(pose_t[0] + 1, pose_t[-1] - seconds - 1, 0.25):
        sel = (pose_t >= start) & (pose_t <= start + seconds)
        p = pose_xyz[sel, :2]
        if len(p) < 10:
            continue
        span = np.linalg.norm(p[-1] - p[0])
        path = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
        score = span / seconds - 2.0 * (path - span)
        if best is None or score > best[0]:
            best = (score, start, span / seconds)
    _, start, speed = best
    sel = (low_t >= start) & (low_t < start + seconds)
    return low_t[sel] - low_t[sel][0], low_q[sel], float(start - pose_t[0]), float(speed)


def render_detour(args, detour, model, data, renderer, cam, opt, overlay, ff, xyz, low_t, low_q, pose_t, pose_xyz,
                  trail, floor_z, fps):
    import mujoco
    clip_t, clip_q, clip_start, speed = gait_clip(low_t, low_q, pose_t, pose_xyz)
    route = np.asarray(detour['routes']['detour']['xy_map3d'], float)
    original = np.asarray(detour['routes']['original']['xy_map3d'], float)
    path, length = resample_polyline(route, speed / fps)
    heading = smooth_heading(np.arctan2(*np.gradient(path, axis=0)[:, ::-1].T), 0.5 * fps)
    z = float(np.median(xyz[:, 2]))
    period = clip_t[-1]
    banner = 'МОДЕЛЬНЫЙ СЦЕНАРИЙ #212 — не измерение'
    lines = ['Модельный объезд препятствия на карте обхода',
             'ящик 0,5 × 0,5 м задан вручную, робот его не видел; путь A* с запасом 0,48 м',
             f'походка — повтор измеренного фрагмента /lowstate ({ru(clip_start)}–{ru(clip_start + period)} с)',
             f'объезд {ru(detour["routes"]["detour"]["check"]["length_m"], 2)} м против {ru(detour["routes"]["original"]["check"]["length_m"], 2)} м по исходному пути']
    legend = [((0, 120, 214, 255), 'модельный объезд #212'),
              ((150, 150, 155, 255), 'исходный путь через ящик'),
              ((242, 158, 26, 255), 'тестовое препятствие 0,5 × 0,5 м')]
    frames = int(args.title_hold * fps) + len(path)
    cam_yaw = smooth_heading(heading, 1.2 * fps)
    for k in range(frames):
        i = max(0, k - int(args.title_hold * fps))
        joints = clip_q[nearest_index(clip_t, np.array([(i / fps) % period]))[0]]
        set_robot(data, np.r_[path[i], z], Rotation.from_euler('z', heading[i]), joints)
        mujoco.mj_forward(model, data)
        center = np.r_[np.mean(route, axis=0), floor_z]
        chase_camera(cam, center + [0, 0, 0.2], cam_yaw[0] + 0.9, 6.2, -48)
        renderer.update_scene(data, cam, opt)
        add_polyline(renderer.scene, original, floor_z + 0.01, 0.02, (0.55, 0.55, 0.58, 1), dashed=True)
        add_polyline(renderer.scene, route, floor_z + 0.012, 0.025, (*DETOUR_RGB, 0.45))
        add_polyline(renderer.scene, path[:i + 1], floor_z + 0.014, 0.03, (*DETOUR_RGB, 1))
        frame = overlay.draw(renderer.render(), lines, trail, path[i], heading[i], detour_xy=path[:i + 1],
                             banner=banner, legend=legend)
        ff.stdin.write(frame.tobytes())
    return dict(source=str(args.detour), speed_m_s=round(speed, 3), length_m=round(float(length), 3),
                gait_clip_from_inspection_s=[round(clip_start, 2), round(clip_start + period, 2)],
                frames=frames, height_m=round(z, 3), note='model scenario, not a measurement')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, default=Path('out/go2-release'),
                   help='unpacked go2-live-full-a5ad606.zip: bag/ and map.npz')
    p.add_argument('--inspection', type=Path, default=Path('data/official/inspection'))
    p.add_argument('--model', type=Path, default=Path('out/issue213/menagerie/unitree_go2/go2.xml'))
    p.add_argument('--detour', type=Path, default=Path('docs/go2-obstacle-demo.json'),
                   help='route.json of #212; pass an empty string to skip the model scenario')
    p.add_argument('--out', type=Path, default=Path('out/issue213/show'))
    p.add_argument('--name', default='go2-show3d-a5ad606.mp4')
    p.add_argument('--pose-topic', default='/mgts/pose3d')
    p.add_argument('--size', type=int, nargs=2, default=(1920, 1080))
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--crf', type=int, default=18)
    p.add_argument('--seconds', type=float, help='render only the first N seconds of the inspection (preview)')
    p.add_argument('--still', type=float, nargs='*', default=[52.0], help='inspection seconds to save as PNG')
    p.add_argument('--wall-cut', type=float, default=0.5, help='wall height above the floor, m')
    p.add_argument('--swept-radius', type=float, default=0.3, help='hide map points this close to published poses, m')
    p.add_argument('--distance', type=float, default=3.6)
    p.add_argument('--elevation', type=float, default=-58)
    p.add_argument('--intro', type=float, default=3.0)
    p.add_argument('--outro', type=float, default=1.5)
    p.add_argument('--title-hold', type=float, default=2.0)
    p.add_argument('--font-dir', type=Path)
    args = p.parse_args(argv)
    if args.detour == Path(''):
        args.detour = None
    os.environ.setdefault('MUJOCO_GL', 'egl')
    print(json.dumps(render(args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
