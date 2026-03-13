# runner_ros2.py

import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import rclpy
import rclpy.duration
import tf2_ros
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from omegaconf import OmegaConf
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from applications.utils.ros_publisher import ROSPublisher
from applications.utils.runner_ros_base import RunnerROSBase
from dualmap.core import Dualmap
from utils.logging_helper import setup_logging


class RunnerROS2(Node, RunnerROSBase):
    """
    ROS2-specific runner. Uses rclpy and ROS2 message_filters for synchronization,
    subscription, and publishing.
    """

    def __init__(self, cfg):
        Node.__init__(self, "runner_ros")
        setup_logging(output_path=cfg.output_path, config_path=cfg.logging_config)
        self.logger = logging.getLogger(__name__)
        self.logger.info("[Runner ROS2]")
        self.logger.info(OmegaConf.to_yaml(cfg))

        self.cfg = cfg
        self.dualmap = Dualmap(cfg)
        RunnerROSBase.__init__(self, cfg, self.dualmap)

        self.bridge = CvBridge()
        self.dataset_cfg = OmegaConf.load(cfg.ros_stream_config_path)
        self.intrinsics = self.load_intrinsics(self.dataset_cfg)
        self.extrinsics = self.load_extrinsics(self.dataset_cfg)

        # Topic Subscribers
        if self.cfg.use_compressed_topic:
            self.logger.warning("[Main] Using compressed topics.")
            self.rgb_sub = Subscriber(
                self, CompressedImage, self.dataset_cfg.ros_topics.rgb
            )
            self.depth_sub = Subscriber(
                self, CompressedImage, self.dataset_cfg.ros_topics.depth
            )
        else:
            self.logger.warning("[Main] Using uncompressed topics.")
            self.rgb_sub = Subscriber(self, Image, self.dataset_cfg.ros_topics.rgb)
            self.depth_sub = Subscriber(self, Image, self.dataset_cfg.ros_topics.depth)

        # Detect pose mode: Odometry topic vs TF lookup
        self.use_tf_mode = not hasattr(self.dataset_cfg.ros_topics, 'odom')

        if self.use_tf_mode:
            # TF mode: obtain camera pose from tf2 transforms
            self.logger.warning("[Main] No 'odom' topic defined. Using TF mode for camera pose.")
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self.tf_source_frame = self.dataset_cfg.tf_source_frame
            self.tf_target_frame = self.dataset_cfg.tf_target_frame
            self.tf_mode_queue = deque()
            self.logger.warning(
                f"[Main] TF lookup: '{self.tf_source_frame}' -> '{self.tf_target_frame}'"
            )

            # Sync only RGB + Depth (no Odometry)
            self.sync = ApproximateTimeSynchronizer(
                [self.rgb_sub, self.depth_sub],
                queue_size=10,
                slop=self.cfg.sync_threshold,
            )
            self.sync.registerCallback(self.synced_callback_tf)
        else:
            # Odometry mode: original behaviour
            self.logger.warning("[Main] Using Odometry topic for camera pose.")
            self.odom_sub = Subscriber(
                self, Odometry, self.dataset_cfg.ros_topics.odom
            )

            # Sync RGB + Depth + Odom
            self.sync = ApproximateTimeSynchronizer(
                [self.rgb_sub, self.depth_sub, self.odom_sub],
                queue_size=10,
                slop=self.cfg.sync_threshold,
            )
            self.sync.registerCallback(self.synced_callback)

        # CameraInfo fallback
        self.create_subscription(
            CameraInfo,
            self.dataset_cfg.ros_topics.camera_info,
            self.camera_info_callback,
            10,
        )

        # Publisher and timer
        self.publisher = ROSPublisher(self, cfg)
        self.publish_executor = ThreadPoolExecutor(max_workers=2)

        timer_period = 1.0 / self.cfg.ros_rate
        self.timer = self.create_timer(timer_period, self.run)

    def synced_callback(self, rgb_msg, depth_msg, odom_msg):
        """Callback for synced RGB-D-Odom input (Odometry mode)."""
        timestamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9

        if self.cfg.use_compressed_topic:
            rgb_img = self.decompress_image(rgb_msg.data, is_depth=False)
            depth_img = self.decompress_image(depth_msg.data, is_depth=True)
        else:
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        depth_factor = getattr(self.dataset_cfg, 'depth_factor', 1000.0)
        depth_img = self.process_depth_image(depth_img, depth_factor)

        translation = np.array(
            [
                odom_msg.pose.pose.position.x,
                odom_msg.pose.pose.position.y,
                odom_msg.pose.pose.position.z,
            ]
        )
        quaternion = np.array(
            [
                odom_msg.pose.pose.orientation.x,
                odom_msg.pose.pose.orientation.y,
                odom_msg.pose.pose.orientation.z,
                odom_msg.pose.pose.orientation.w,
            ]
        )

        pose_matrix = self.build_pose_matrix(translation, quaternion)
        self.push_data(rgb_img, depth_img, pose_matrix, timestamp)
        self.last_message_time = self.get_clock().now().nanoseconds / 1e9

    def synced_callback_tf(self, rgb_msg, depth_msg):
        """Callback for synced RGB-D input using TF for camera pose. Buffers frames."""
        if len(self.tf_mode_queue) > 50:
            self.tf_mode_queue.popleft()
        self.tf_mode_queue.append((rgb_msg, depth_msg, 0))

    def process_tf_queue(self):
        """Process buffered frames and resolve TF transforms without blocking."""
        while self.tf_mode_queue:
            rgb_msg, depth_msg, retries = self.tf_mode_queue[0]
            stamp = rgb_msg.header.stamp

            # Look up the transform at the image timestamp
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.tf_source_frame,
                    self.tf_target_frame,
                    stamp,
                    timeout=rclpy.duration.Duration(seconds=0.0),
                )
            except Exception as e:
                # Wait up to sync_threshold seconds (based on ros_rate) for the TF to arrive
                max_retries = int(self.cfg.ros_rate * self.cfg.sync_threshold)
                if retries < max_retries:
                    self.tf_mode_queue[0] = (rgb_msg, depth_msg, retries + 1)
                    break  # Break and wait for next timer tick
                else:
                    self.logger.warning(
                        f"[Main] Dropping frame after persistent TF failures ({max_retries} retries): {e}"
                    )
                    self.tf_mode_queue.popleft()
                    continue

            # Success: remove from queue
            self.tf_mode_queue.popleft()

            # Extract translation and quaternion from TransformStamped
            t = transform.transform.translation
            r = transform.transform.rotation
            translation = np.array([t.x, t.y, t.z])
            quaternion = np.array([r.x, r.y, r.z, r.w])

            # Process images (identical to Odometry mode)
            if self.cfg.use_compressed_topic:
                rgb_img = self.decompress_image(rgb_msg.data, is_depth=False)
                depth_img = self.decompress_image(depth_msg.data, is_depth=True)
            else:
                rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
                depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

            depth_factor = getattr(self.dataset_cfg, 'depth_factor', 1000.0)
            depth_img = self.process_depth_image(depth_img, depth_factor)

            timestamp = stamp.sec + stamp.nanosec * 1e-9
            pose_matrix = self.build_pose_matrix(translation, quaternion)
            self.push_data(rgb_img, depth_img, pose_matrix, timestamp)
            self.last_message_time = self.get_clock().now().nanoseconds / 1e9

    def camera_info_callback(self, msg):
        """Populate intrinsics from CameraInfo topic if not already loaded."""
        if self.intrinsics is None:
            self.intrinsics = np.array(msg.k).reshape(3, 3)
            self.logger.warning("[Main] Camera intrinsics received and stored.")

    def run(self):
        """Periodic processing loop triggered by ROS2 timer."""
        if getattr(self, "use_tf_mode", False):
            self.process_tf_queue()

        self.run_once(lambda: self.get_clock().now().nanoseconds / 1e9)
        self.publish_executor.submit(self.publisher.publish_all, self.dualmap)

    def shutdown_all_threads(self):
        """Clean up all threads and timers."""
        self.logger.warning("[Main] Shutting down all threads and timers.")
        try:
            self.timer.cancel()
        except Exception as e:
            self.logger.warning(f"[Main] Failed to cancel timer: {e}")
        self.publish_executor.shutdown(wait=True)

    def destroy_node(self):
        """Override base destroy_node with cleanup logic."""
        if not self.shutdown_requested:
            self.logger.warning("[Main] Shutting down abruptly. Triggering end_process() to save maps.")
            self.dualmap.end_process()
            self.shutdown_requested = True
            
        self.shutdown_all_threads()
        super().destroy_node()


def run_ros2(cfg):
    """Entry point for launching ROS2 runner."""
    rclpy.init()
    runner = RunnerROS2(cfg)
    runner.logger.warning("[Main] ROS2 Runner started. Waiting for data stream...")
    try:
        while rclpy.ok() and not runner.shutdown_requested:
            rclpy.spin_once(runner, timeout_sec=0.1)
    except KeyboardInterrupt:
        runner.logger.warning("[Main] KeyboardInterrupt received. Shutting down.")
    finally:
        runner.destroy_node()
        rclpy.shutdown()
        runner.logger.warning("[Main] Done.")
