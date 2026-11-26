"""ROS 2 node: real-time localization of the Unitree Go2 on a prebuilt 3D map.

Subscribes
  /utlidar/cloud_base   sensor_msgs/PointCloud2 (base_link, per-point ``time``)
  /utlidar/robot_odom   nav_msgs/Odometry (odom -> base_link)
  /initialpose          geometry_msgs/PoseWithCovarianceStamped (RViz "2D Pose Estimate")
Publishes
  TF map->odom, odom->base_link (the bags carry no TF), base_link->utlidar_lidar
  /mgts/pose            geometry_msgs/PoseWithCovarianceStamped (map)
  /mgts/odometry        nav_msgs/Odometry (map -> base_link)
  /mgts/path            nav_msgs/Path, localized trajectory
  /mgts/odom_path       nav_msgs/Path, raw odometry placed with the initial correction (drift comparison)
  /mgts/map             sensor_msgs/PointCloud2, the 3D map (transient local)
  /mgts/scan_map        sensor_msgs/PointCloud2, current deskewed sweep in the map frame
  /mgts/dynamic_points  sensor_msgs/PointCloud2, sweep points farther than ``dynamic_gate`` from the map
  /mgts/status          std_msgs/String, JSON with status, quality and timing of the last sweep
  /mgts/timing          std_msgs/Float64MultiArray [deskew, prepare, icp, total, latency] ms

Time: the organizers' bags have header stamps 80.8 s behind the recording
clock. With ``stamp_mode: auto`` and use_sim_time the node measures that offset
on the first odometry message and stamps its outputs on the /clock time base,
so RViz and TF lookups work during ``ros2 bag play --clock``.
"""
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
import rclpy.executors
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float64MultiArray, String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from .core import cloud, se3
from .core.deskew import deskew
from .core.icp import PointMap, residuals_to_map
from .core.localizer import Localizer, LocalizerParams, MATCHED
from .core.mapio import read_pcd, read_poses
from .core.odom_buffer import OdomBuffer

# Estimated from the bag: cloud_base = R * cloud + t exactly (residual 1e-8 m), i.e. lidar pitched 165 deg about y.
LIDAR_IN_BASE = se3.make(se3.rpy_to_rot(0.0, math.radians(165.012), 0.0), (0.2822, 0.0, -0.0247))


def to_time(stamp):
    sec = int(math.floor(stamp))
    return Time(sec=sec, nanosec=int(round((stamp - sec) * 1e9)) % 1000000000)


def stamp_of(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def make_cloud(points, frame, stamp, intensity=None):
    msg = PointCloud2()
    msg.header.frame_id = frame
    msg.header.stamp = to_time(stamp)
    n = len(points)
    fields = [PointField(name=c, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, c in enumerate('xyz')]
    cols = [np.asarray(points, dtype=np.float32)]
    if intensity is not None:
        fields.append(PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1))
        cols.append(np.asarray(intensity, dtype=np.float32).reshape(-1, 1))
    data = np.hstack(cols).astype(np.float32)
    msg.fields = fields
    msg.height = 1
    msg.width = n
    msg.point_step = 4 * data.shape[1] if n else 4 * len(fields)
    msg.row_step = msg.point_step * n
    msg.is_bigendian = False
    msg.is_dense = True
    msg.data = data.tobytes()
    return msg


def fill_pose(pose_msg, t):
    q = se3.rot_to_quat(t[:3, :3])
    pose_msg.position.x, pose_msg.position.y, pose_msg.position.z = (float(v) for v in t[:3, 3])
    pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w = (float(v) for v in q)


def make_tf(parent, child, t, stamp):
    m = TransformStamped()
    m.header.frame_id = parent
    m.header.stamp = to_time(stamp)
    m.child_frame_id = child
    m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = (float(v) for v in t[:3, 3])
    q = se3.rot_to_quat(t[:3, :3])
    m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w = (float(v) for v in q)
    return m


class LocalizerNode(Node):
    def __init__(self):
        super().__init__('mgts_localizer')
        dp = LocalizerParams()
        P = self.declare_parameter
        self.map_dir = Path(P('map_dir', '/ros2_ws/mgts_map').value)
        self.map_file = P('map_file', 'map.pcd').value
        self.cloud_topic = P('cloud_topic', '/utlidar/cloud_base').value
        self.odom_topic = P('odom_topic', '/utlidar/robot_odom').value
        self.map_frame = P('map_frame', 'map').value
        self.odom_frame = P('odom_frame', 'odom').value
        self.base_frame = P('base_frame', 'base_link').value
        self.init_mode = P('init_mode', 'odom').value              # odom | search | pose
        self.initial_pose = list(P('initial_pose', [0.0, 0.0, 0.0]).value)
        self.stamp_mode = P('stamp_mode', 'auto').value            # auto | header | clock
        self.publish_odom_tf = P('publish_odom_tf', True).value
        self.publish_lidar_tf = P('publish_lidar_tf', True).value
        self.deskew_enabled = P('deskew', True).value
        self.dynamic_gate = P('dynamic_gate', 0.3).value
        self.max_pending = P('max_pending', 2).value
        self.odom_wait = P('odom_wait', 0.15).value
        self.timing_csv = P('timing_csv', '').value
        self.path_every = P('path_every', 7).value               # sweeps between Path messages (~2 Hz)
        lp = LocalizerParams(
            reg_voxel=P('reg_voxel', dp.reg_voxel).value,
            min_range=P('min_range', dp.min_range).value,
            max_range=P('max_range', dp.max_range).value,
            prior_sigma_rot=P('prior_sigma_rot', dp.prior_sigma_rot).value,
            prior_sigma_trans=P('prior_sigma_trans', dp.prior_sigma_trans).value,
            planar=P('planar', dp.planar).value,
            min_inlier_ratio=P('min_inlier_ratio', dp.min_inlier_ratio).value,
            min_structure_ratio=P('min_structure_ratio', dp.min_structure_ratio).value,
            floor_height=P('floor_height', dp.floor_height).value,
            max_correction_m=P('max_correction_m', dp.max_correction_m).value,
            max_iterations=P('max_iterations', dp.max_iterations).value,
            lost_after=P('lost_after', dp.lost_after).value,
        )

        t0 = time.perf_counter()
        pts, inten = read_pcd(self.map_dir / self.map_file)
        self.target = PointMap(pts)
        places = None
        traj = self.map_dir / 'mapping_trajectory.csv'
        if traj.exists():
            places = read_poses(traj)[1]
        self.loc = Localizer(self.target, lp, places)
        self.get_logger().info('map %s: %d points, KD-tree and normals in %.2f s'
                               % (self.map_dir / self.map_file, len(pts), time.perf_counter() - t0))

        self.odom = OdomBuffer(horizon=30.0)
        self.pending = deque()
        self.offset = None              # output stamp = header stamp + offset
        self.dropped = 0
        self.processed = 0
        self.rows = []
        self.t_map_odom0 = None
        self.last_odom_msg = None

        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        odom_qos = QoSProfile(depth=300, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Odometry, self.odom_topic, self.on_odom, odom_qos)
        self.create_subscription(PointCloud2, self.cloud_topic, self.on_cloud, sensor_qos)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.on_initialpose, 10)
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, '/mgts/pose', 10)
        self.pub_odom = self.create_publisher(Odometry, '/mgts/odometry', 10)
        self.pub_path = self.create_publisher(PathMsg, '/mgts/path', latched)
        self.pub_opath = self.create_publisher(PathMsg, '/mgts/odom_path', latched)
        self.pub_map = self.create_publisher(PointCloud2, '/mgts/map', latched)
        self.pub_scan = self.create_publisher(PointCloud2, '/mgts/scan_map', 10)
        self.pub_dyn = self.create_publisher(PointCloud2, '/mgts/dynamic_points', 10)
        self.pub_status = self.create_publisher(String, '/mgts/status', 10)
        self.pub_timing = self.create_publisher(Float64MultiArray, '/mgts/timing', 10)
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        self.path = PathMsg()
        self.path.header.frame_id = self.map_frame
        self.opath = PathMsg()
        self.opath.header.frame_id = self.map_frame
        self.map_msg = make_cloud(pts, self.map_frame, 0.0, inten)
        self.map_published = False
        self.create_timer(0.02, self.process_pending)

    # ------------------------------------------------------------ time base
    def out_stamp(self, header_stamp):
        return header_stamp + (self.offset or 0.0)

    def _init_offset(self, header_stamp):
        """Fix the output time base on the first odometry message. Returns False while undecided."""
        if self.offset is not None:
            return True
        now = self.get_clock().now().nanoseconds * 1e-9
        use_sim = self.get_parameter('use_sim_time').value
        if self.stamp_mode == 'header':
            self.offset = 0.0
        elif self.stamp_mode == 'clock' or (self.stamp_mode == 'auto' and use_sim):
            if now <= 0.0:
                # /clock has not arrived yet: decide on a later message, never on time zero
                self.get_logger().info('waiting for /clock to fix the output time base',
                                       throttle_duration_sec=2.0)
                return False
            self.offset = (now - header_stamp) if abs(now - header_stamp) > 1.0 else 0.0
        else:
            self.offset = 0.0
        self.get_logger().info('output stamps = header + %.3f s (stamp_mode=%s, use_sim_time=%s)'
                               % (self.offset, self.stamp_mode, use_sim))
        return True

    # ------------------------------------------------------------ callbacks
    def on_odom(self, msg):
        st = stamp_of(msg.header)
        ready = self._init_offset(st)
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        t = se3.from_pose((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))
        if not self.odom.add(st, t):
            return
        self.last_odom_msg = msg
        if not ready:
            return
        if self.publish_odom_tf:
            self.tf.sendTransform(make_tf(self.odom_frame, self.base_frame, t, self.out_stamp(st)))
        if self.loc.initialized:
            # keep map->odom fresh between sweeps for consumers that look up "now"
            self.tf.sendTransform(make_tf(self.map_frame, self.odom_frame, self.loc.t_map_odom, self.out_stamp(st)))

    def on_cloud(self, msg):
        self.pending.append((time.perf_counter(), msg))
        while len(self.pending) > self.max_pending:
            self.pending.popleft()
            self.dropped += 1

    def on_initialpose(self, msg):
        latest = self.odom.at(self.odom.times[-1]) if len(self.odom) else None
        if latest is None:
            self.get_logger().warn('initial pose ignored: no odometry yet')
            return
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        t_map_base = se3.from_pose((p.x, p.y, latest[2, 3]), (q.x, q.y, q.z, q.w))
        self.loc.set_pose(t_map_base, latest)
        self.get_logger().info('initial pose set from /initialpose: x=%.2f y=%.2f yaw=%.1f deg'
                               % (p.x, p.y, math.degrees(se3.yaw_of(t_map_base))))

    # ------------------------------------------------------------ processing
    def process_pending(self):
        if not self.map_published and self.offset is not None:
            self.map_msg.header.stamp = to_time(self.out_stamp(self.odom.times[-1]) if len(self.odom) else 0.0)
            self.pub_map.publish(self.map_msg)
            self.static_lidar()
            self.map_published = True
        while self.pending and self.offset is not None:
            arrived, msg = self.pending[0]
            st = stamp_of(msg.header)
            f = cloud.decode(msg)
            times = f.get('time', np.zeros(len(f['x'])))
            end = st + (float(times.max()) if len(times) else 0.0)
            newest = self.odom.times[-1] if len(self.odom) else -1e18
            if newest < end and time.perf_counter() - arrived < self.odom_wait:
                return  # wait for odometry covering the whole sweep
            self.pending.popleft()
            self.process_sweep(msg, f, times, st, arrived)

    def static_lidar(self):
        if self.publish_lidar_tf:
            self.static_tf.sendTransform(make_tf(self.base_frame, 'utlidar_lidar', LIDAR_IN_BASE,
                                                 self.out_stamp(self.odom.times[-1]) if len(self.odom) else 0.0))

    def process_sweep(self, msg, f, times, st, arrived):
        t0 = time.perf_counter()
        pts = cloud.xyz(f)
        keep = cloud.valid_mask(pts)
        pts, times = pts[keep], times[keep]
        try:
            if self.deskew_enabled:
                pts = deskew(pts, times, st, self.odom, max_extrapolation=0.15)
            t_ob = self.odom.at(st, max_extrapolation=0.15)
            if t_ob is None:
                raise LookupError('no odometry at %.3f' % st)
        except LookupError as e:
            self.get_logger().warn('sweep %.3f skipped: %s' % (st, e), throttle_duration_sec=2.0)
            return
        deskew_ms = (time.perf_counter() - t0) * 1e3

        if not self.loc.initialized:
            if self.init_mode == 'odom':
                self.loc.set_map_odom(np.eye(4))
            elif self.init_mode == 'pose':
                x, y, yaw = self.initial_pose
                self.loc.set_map_odom(se3.make(se3.rot_z(yaw), (x, y, 0.0)) @ se3.inv(se3.planar(t_ob)))
        r = self.loc.step(st, pts, t_ob)
        if self.t_map_odom0 is None and self.loc.initialized:
            self.t_map_odom0 = r.t_map_odom.copy()

        out = self.out_stamp(st)
        if self.loc.initialized:
            self.tf.sendTransform(make_tf(self.map_frame, self.odom_frame, r.t_map_odom, out))
        self.publish_pose(r, out, t_ob)
        in_map = se3.apply(r.t_map_base, pts)
        self.pub_scan.publish(make_cloud(in_map, self.map_frame, out))
        res = residuals_to_map(in_map, self.target, gate=self.dynamic_gate)
        self.pub_dyn.publish(make_cloud(in_map[~np.isfinite(res)], self.map_frame, out))
        latency = (time.perf_counter() - arrived) * 1e3
        total = (time.perf_counter() - t0) * 1e3
        tim = r.timings_ms
        self.pub_timing.publish(Float64MultiArray(data=[deskew_ms, tim.get('prepare', 0.0), tim.get('icp', 0.0),
                                                        total, latency]))
        sr = None if math.isnan(r.structure_ratio) else round(r.structure_ratio, 3)
        status = {'stamp': round(st, 6), 'status': r.status, 'inlier_ratio': round(r.inlier_ratio, 3),
                  'structure_ratio': sr,
                  'rmse': None if math.isnan(r.rmse) else round(r.rmse, 4), 'points': r.n_points,
                  'correction_m': round(r.correction_m, 3), 'deskew_ms': round(deskew_ms, 2),
                  'icp_ms': round(tim.get('icp', float('nan')), 2), 'total_ms': round(total, 2),
                  'latency_ms': round(latency, 2), 'dropped': self.dropped}
        self.pub_status.publish(String(data=json.dumps(status)))
        self.processed += 1
        self.rows.append((st, r.status, *r.t_map_base[:3, 3], se3.yaw_of(r.t_map_base), r.inlier_ratio,
                          r.structure_ratio, r.rmse, r.n_points, deskew_ms, tim.get('prepare', float('nan')),
                          tim.get('icp', float('nan')), tim.get('search', float('nan')),
                          deskew_ms + tim.get('total', float('nan')), total, latency))
        if self.processed % 50 == 0:
            self.get_logger().info('sweeps %d, status %s, inliers %.2f, total %.1f ms, latency %.1f ms, dropped %d'
                                   % (self.processed, r.status, r.inlier_ratio, total, latency, self.dropped))

    def publish_pose(self, r, out, t_ob):
        pm = PoseWithCovarianceStamped()
        pm.header.frame_id = self.map_frame
        pm.header.stamp = to_time(out)
        fill_pose(pm.pose.pose, r.t_map_base)
        good = r.status == MATCHED
        sxy, syaw = (0.03, 0.01) if good else (0.3, 0.1)
        cov = np.zeros(36)
        cov[0] = cov[7] = sxy ** 2
        cov[14] = 0.01 ** 2
        cov[21] = cov[28] = 0.01 ** 2
        cov[35] = syaw ** 2
        pm.pose.covariance = cov.tolist()
        self.pub_pose.publish(pm)

        od = Odometry()
        od.header = pm.header
        od.child_frame_id = self.base_frame
        od.pose = pm.pose
        if self.last_odom_msg is not None:
            od.twist = self.last_odom_msg.twist
        self.pub_odom.publish(od)

        # every sweep goes into the paths; the growing messages are published at a lower rate
        ps = PoseStamped()
        ps.header = pm.header
        fill_pose(ps.pose, r.t_map_base)
        self.path.poses.append(ps)
        self.path.header.stamp = pm.header.stamp
        if self.t_map_odom0 is not None:
            po = PoseStamped()
            po.header = pm.header
            fill_pose(po.pose, self.t_map_odom0 @ t_ob)
            self.opath.poses.append(po)
            self.opath.header.stamp = pm.header.stamp
        if self.processed % max(1, self.path_every) == 0:
            self.pub_path.publish(self.path)
            self.pub_opath.publish(self.opath)

    def write_timings(self):
        if not self.timing_csv or not self.rows:
            return
        path = Path(self.timing_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as fh:
            fh.write('stamp,status,x,y,z,yaw,inlier_ratio,structure_ratio,rmse,n_points,deskew_ms,prepare_ms,icp_ms,search_ms,'
                     'localization_ms,total_ms,latency_ms\n')
            for row in self.rows:
                fh.write(','.join(str(v) if isinstance(v, str) else '%.6f' % v for v in row) + '\n')
        loc_ms = np.array([r[-3] for r in self.rows], dtype=float)
        arr = np.array([r[-2] for r in self.rows], dtype=float)
        lat = np.array([r[-1] for r in self.rows], dtype=float)
        summary = {'sweeps': len(self.rows), 'dropped': self.dropped,
                   'status_counts': {s: sum(1 for r in self.rows if r[1] == s) for s in {r[1] for r in self.rows}},
                   'localization_ms': {'p50': float(np.nanpercentile(loc_ms, 50)),
                                       'p95': float(np.nanpercentile(loc_ms, 95)), 'max': float(np.nanmax(loc_ms))},
                   'total_ms': {'p50': float(np.nanpercentile(arr, 50)), 'p95': float(np.nanpercentile(arr, 95)),
                                'max': float(np.nanmax(arr))},
                   'latency_ms': {'p50': float(np.nanpercentile(lat, 50)), 'p95': float(np.nanpercentile(lat, 95)),
                                  'max': float(np.nanmax(lat))}}
        path.with_suffix('.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        self.get_logger().info('timings written to %s' % path)


def main(args=None):
    rclpy.init(args=args)
    node = LocalizerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.write_timings()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
