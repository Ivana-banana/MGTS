"""Display the ROS 2 result poses as a path and a TF frame for the RViz camera."""
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class ReplayView(Node):
    def __init__(self):
        super().__init__('mgts_replay_view')
        self.path = Path()
        self.publisher = self.create_publisher(Path, '/mgts/display_path', 1)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(PoseStamped, '/mgts/pose3d', self.receive, 10)

    def receive(self, pose):
        stamp = pose.header.stamp
        previous = self.path.header.stamp
        if (stamp.sec, stamp.nanosec) < (previous.sec, previous.nanosec):
            self.path.poses.clear()
        self.path.header = pose.header
        self.path.poses.append(pose)
        self.publisher.publish(self.path)
        transform = TransformStamped(header=pose.header, child_frame_id='go2_view')
        p = pose.pose.position
        transform.transform.translation.x = p.x
        transform.transform.translation.y = p.y
        transform.transform.translation.z = p.z
        transform.transform.rotation = pose.pose.orientation
        self.tf.sendTransform(transform)


def main():
    rclpy.init()
    node = ReplayView()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

