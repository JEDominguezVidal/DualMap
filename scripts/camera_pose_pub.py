#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class CameraPosePublisher(Node):

    def __init__(self):
        super().__init__('camera_pose_publisher')
        self.publisher = self.create_publisher(
            Odometry,
            '/camera/pose',
            10
        )
        self.timer = self.create_timer(1.0 / 30.0, self.publish_msg)

    def publish_msg(self):
        msg = Odometry()

        # Timestamp actual de ROS (no del sistema directamente)
        msg.header.stamp = self.get_clock().now().to_msg()

        # Opcional pero recomendable
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraPosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

