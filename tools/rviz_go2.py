"""Show the published Go2 poses and observed wall surfaces in RViz."""

import argparse
import sys
from collections import defaultdict
from math import atan2, cos, sin
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.utilities import remove_ros_args
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


VOXEL_M = 0.12  # Resolution used to create the published Go2 map.
WALL_BASE_M = 0.0  # Excludes the floor layer in this map frame.
MIN_VERTICAL_CELLS = 4
FACE_COLORS = ((0.42, 0.55, 0.65), (0.51, 0.65, 0.75), (0.71, 0.82, 0.89))


def wall_voxels(points):
    """Select observed upright runs with neighboring support at the same height."""
    columns = defaultdict(set)
    for x, y, z in np.floor(points / VOXEL_M).astype(int):
        if z * VOXEL_M >= WALL_BASE_M:
            columns[x, y].add(z)

    runs = set()
    for (x, y), levels in columns.items():
        for bottom in levels:
            if bottom - 1 in levels:
                continue
            top = bottom + 1
            while top in levels:
                top += 1
            if top - bottom >= MIN_VERTICAL_CELLS:
                runs.update((x, y, z) for z in range(bottom, top))

    neighbors = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    return {(x, y, z) for x, y, z in runs
            if any((x + dx, y + dy, z) in runs for dx, dy in neighbors)}


def wall_triangles(cells, top_m):
    """Mesh the occupied shell. Greedily merge coplanar faces without filling gaps."""
    cells = {cell for cell in cells if cell[2] * VOXEL_M < top_m}
    planes = defaultdict(set)
    for cell in cells:
        for axis in range(3):
            u, v = (axis + 1) % 3, (axis + 2) % 3
            for side in (-1, 1):
                neighbor = list(cell)
                neighbor[axis] += side
                if tuple(neighbor) not in cells:
                    planes[axis, side, cell[axis] + (side == 1)].add((cell[u], cell[v]))

    triangles = [[], [], []]
    for (axis, side, offset), squares in sorted(planes.items()):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for left, bottom in sorted(squares):
            if (left, bottom) not in squares:
                continue
            right, top = left + 1, bottom + 1
            while (right, bottom) in squares:
                right += 1
            while all((i, top) in squares for i in range(left, right)):
                top += 1
            squares.difference_update((i, j) for i in range(left, right) for j in range(bottom, top))

            corners = []
            for a, b in ((left, bottom), (right, bottom), (right, top), (left, top)):
                corner = [0.0, 0.0, 0.0]
                corner[axis], corner[u], corner[v] = offset * VOXEL_M, a * VOXEL_M, b * VOXEL_M
                corner[2] = min(corner[2], top_m)
                corners.append(tuple(corner))
            if side < 0:
                corners.reverse()
            triangles[axis].extend(corners[i] for i in (0, 1, 2, 0, 2, 3))
    return triangles


def marker(namespace, idx, shape, rgb, *, frame='map'):
    result = Marker()
    result.header.frame_id = frame
    result.ns, result.id, result.type = namespace, idx, shape
    result.pose.orientation.w = 1.0
    result.scale.x = result.scale.y = result.scale.z = 1.0
    result.color.r, result.color.g, result.color.b = rgb
    result.color.a = 1.0
    return result


def walls_message(triangles, alpha):
    markers = []
    for axis, vertices in enumerate(triangles):
        surface = marker('walls', axis, Marker.TRIANGLE_LIST, FACE_COLORS[axis])
        surface.color.a = alpha
        surface.points = [Point(x=x, y=y, z=z) for x, y, z in vertices]
        markers.append(surface)
    return MarkerArray(markers=markers)


def robot_markers():
    """Procedural Go2 silhouette with a fixed illustrative stance in base_link."""
    body = 0.94, 0.67, 0.28
    shell = 0.66, 0.73, 0.78
    dark = 0.16, 0.22, 0.27
    joint = 0.28, 0.33, 0.40
    headlight = 0.98, 0.68, 0.16
    identifiers = []

    def part(shape, xyz, scale, color, pitch=0.):
        piece = marker('robot', len(identifiers), shape, color, frame='rviz_go2')
        piece.pose.position.x, piece.pose.position.y, piece.pose.position.z = xyz
        piece.pose.orientation.y = sin(pitch / 2)
        piece.pose.orientation.w = cos(pitch / 2)
        piece.scale.x, piece.scale.y, piece.scale.z = scale
        piece.frame_locked = True
        identifiers.append(piece)

    def link(start, end, width, color):
        a, b = np.array(start), np.array(end)
        dx, _, dz = b - a
        part(Marker.CUBE, tuple((a + b) / 2), (width, width, float(np.hypot(dx, dz))),
             color, atan2(dx, dz))

    part(Marker.CUBE, (0., 0., .035), (.59, .24, .13), body)
    part(Marker.CUBE, (0., 0., -.045), (.54, .22, .065), dark)
    part(Marker.CUBE, (.282, 0., .029), (.105, .20, .11), joint)
    part(Marker.CYLINDER, (.259, 0., .101), (.10, .10, .038), dark)
    for y in (-.056, .056):
        part(Marker.CUBE, (.338, y, .032), (.01, .067, .031), headlight)
    for x in (-.215, .215):
        for y in (-.147, .147):
            hip = x, y, -.044
            knee = x + .085, y, -.205
            toe = x + .014, y, -.314
            part(Marker.SPHERE, hip, (.127, .115, .123), joint)
            link(hip, knee, .072, shell)
            part(Marker.SPHERE, knee, (.079, .079, .079), joint)
            link(knee, toe, .043, dark)
            part(Marker.SPHERE, toe, (.068, .065, .048), dark)
    return MarkerArray(markers=identifiers)


class Go2View(Node):
    def __init__(self, triangles, pose_topic, wall_alpha=1.):
        super().__init__('mgts_rviz_go2')
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.walls = self.create_publisher(MarkerArray, '/rviz_go2/walls', latched)
        self.robot = self.create_publisher(MarkerArray, '/rviz_go2/robot', latched)
        self.trail = self.create_publisher(MarkerArray, '/rviz_go2/trail', latched)
        self.walls.publish(walls_message(triangles, wall_alpha))
        self.robot.publish(robot_markers())
        self.broadcaster = TransformBroadcaster(self)
        self.transform = None
        self.line = marker('trail', 0, Marker.LINE_STRIP, (0.18, 0.95, 0.74))
        self.line.scale.x = .067
        self.last_stamp = None
        self.pose_count = 0
        self.create_subscription(PoseStamped, pose_topic, self.receive_pose, 100)
        self.create_timer(0.1, self.publish_transform)
        self.get_logger().info(f'Треугольников стен: {sum(map(len, triangles)) // 3}. Слушаю {pose_topic}.')

    def receive_pose(self, message):
        position, orientation = message.pose.position, message.pose.orientation
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        if (message.header.frame_id != 'map' or not np.isfinite([position.x, position.y, position.z]).all()
                or not np.isclose(np.linalg.norm(quat), 1., atol=1e-4, rtol=0.)):
            self.get_logger().warning('Пропущена некорректная поза в кадре карты.', throttle_duration_sec=2.)
            return

        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        if self.last_stamp is not None and stamp < self.last_stamp:
            self.line.points.clear()
            self.pose_count = 0
        if stamp == self.last_stamp:
            return
        self.last_stamp = stamp
        self.pose_count += 1

        self.transform = TransformStamped(child_frame_id='rviz_go2')
        self.transform.header.frame_id = 'map'
        self.transform.transform.translation.x = position.x
        self.transform.transform.translation.y = position.y
        self.transform.transform.translation.z = position.z
        self.transform.transform.rotation = orientation
        self.publish_transform()
        self.line.points.append(position)
        self.trail.publish(MarkerArray(markers=[self.line]))

    def publish_transform(self):
        if self.transform is not None:
            self.transform.header.stamp = self.get_clock().now().to_msg()
            self.broadcaster.sendTransform(self.transform)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('map_file', type=Path, help='Файл карты map.npz из выпуска Go2')
    parser.add_argument('--pose-topic', default='/mgts/pose3d')
    parser.add_argument('--wall-top', type=float, default=1.35, help='Верх среза стен по оси Z карты, м')
    parser.add_argument('--wall-alpha', type=float, default=1., help='Непрозрачность стен от 0 до 1')
    args = parser.parse_args(remove_ros_args(args=sys.argv)[1:])
    if not np.isfinite(args.wall_top) or args.wall_top <= WALL_BASE_M:
        parser.error('--wall-top должен быть конечным числом выше нуля')
    if not 0. < args.wall_alpha <= 1.:
        parser.error('--wall-alpha должен быть больше 0 и не больше 1')
    with np.load(args.map_file, allow_pickle=False) as data:
        points = data['xyz']
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        parser.error('Карта должна содержать массив xyz размера N × 3 с конечными координатами')
    triangles = wall_triangles(wall_voxels(points), args.wall_top)
    if not any(triangles):
        parser.error('На карте не найдены стеновые поверхности для указанного среза')

    rclpy.init()
    node = Go2View(triangles, args.pose_topic, args.wall_alpha)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        print(f'Поз на последнем повторе: {node.pose_count}', flush=True)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
