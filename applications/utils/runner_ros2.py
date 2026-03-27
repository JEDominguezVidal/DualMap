# runner_ros2.py

import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
import tf2_ros
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from omegaconf import OmegaConf
from rclpy.executors import ExternalShutdownException
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
        self.use_tf_mode = not hasattr(self.dataset_cfg.ros_topics, "odom")
        self.configure_pose_contract(self.dataset_cfg, self.use_tf_mode)

        if self.use_tf_mode:
            # TF mode: obtain camera pose from tf2 transforms
            self.logger.warning(
                "[Main] No 'odom' topic defined. Using TF mode for camera pose."
            )
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self.tf_world_frame, self.tf_camera_frame = self.resolve_tf_frames(
                self.dataset_cfg
            )
            self.tf_mode_queue = deque()
            self.logger.warning(
                "[Main] TF lookup: '%s' -> '%s'",
                self.tf_world_frame,
                self.tf_camera_frame,
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
        rgb_timestamp = self.stamp_to_seconds(rgb_msg.header.stamp)
        depth_timestamp = self.stamp_to_seconds(depth_msg.header.stamp)
        odom_timestamp = self.stamp_to_seconds(odom_msg.header.stamp)
        timestamp = rgb_timestamp
        pose_frames = self.format_pose_frames(
            odom_msg.header.frame_id,
            odom_msg.child_frame_id,
            default="odom(unknown_frames)",
        )

        should_process, _ = self.evaluate_frame_sync(
            rgb_timestamp=rgb_timestamp,
            depth_timestamp=depth_timestamp,
            pose_timestamp=odom_timestamp,
            pose_mode=self.pose_represents,
            pose_frames=pose_frames,
        )
        if not should_process:
            return

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
        self.push_data(
            rgb_img,
            depth_img,
            pose_matrix,
            timestamp,
            rgb_timestamp=rgb_timestamp,
            depth_timestamp=depth_timestamp,
            pose_timestamp=odom_timestamp,
            pose_mode=self.pose_represents,
        )
        self.last_message_time = self.get_clock().now().nanoseconds / 1e9

    def synced_callback_tf(self, rgb_msg, depth_msg):
        """Callback for synced RGB-D input using TF for camera pose. Buffers frames."""
        queue_limit = (
            self.tf_runtime_queue_size
            if self.has_valid_tf or not self.wait_for_first_valid_tf
            else self.tf_warmup_queue_size
        )
        if queue_limit > 0 and len(self.tf_mode_queue) >= queue_limit:
            old_entry = self.tf_mode_queue.popleft()
            old_rgb = old_entry["rgb_msg"]
            old_depth = old_entry["depth_msg"]
            rgb_timestamp = self.stamp_to_seconds(old_rgb.header.stamp)
            depth_timestamp = self.stamp_to_seconds(old_depth.header.stamp)
            pose_frames = self.format_pose_frames(
                self.tf_world_frame,
                self.tf_camera_frame,
            )
            if self.wait_for_first_valid_tf and not self.has_valid_tf:
                self.log_frame_sync(
                    status="ignored_pre_tf",
                    rgb_depth_dt=abs(rgb_timestamp - depth_timestamp),
                    pose_rgb_dt=None,
                    pose_mode=self.pose_represents,
                    pose_frames=pose_frames,
                    reason=f"tf_warmup_queue_full(limit={queue_limit})",
                )
            else:
                self.dropped_frame_count += 1
                self.log_frame_sync(
                    status="dropped",
                    rgb_depth_dt=abs(rgb_timestamp - depth_timestamp),
                    pose_rgb_dt=None,
                    pose_mode=self.pose_represents,
                    pose_frames=pose_frames,
                    reason=f"tf_runtime_queue_full(limit={queue_limit})",
                )
        self.tf_mode_queue.append(
            {
                "rgb_msg": rgb_msg,
                "depth_msg": depth_msg,
                "enqueue_time": time.monotonic(),
                "last_warning_time": None,
            }
        )

    def process_tf_queue(self):
        """Process buffered frames and resolve TF transforms without blocking."""
        while self.tf_mode_queue:
            entry = self.tf_mode_queue[0]
            rgb_msg = entry["rgb_msg"]
            depth_msg = entry["depth_msg"]
            enqueue_time = entry["enqueue_time"]
            stamp = rgb_msg.header.stamp
            rgb_timestamp = self.stamp_to_seconds(rgb_msg.header.stamp)
            depth_timestamp = self.stamp_to_seconds(depth_msg.header.stamp)
            pose_frames = self.format_pose_frames(
                self.tf_world_frame,
                self.tf_camera_frame,
            )

            if (
                abs(rgb_timestamp - depth_timestamp) > self.max_rgb_depth_dt
                and self.drop_unsynced_frames
            ):
                self.evaluate_frame_sync(
                    rgb_timestamp=rgb_timestamp,
                    depth_timestamp=depth_timestamp,
                    pose_timestamp=rgb_timestamp,
                    pose_mode=self.pose_represents,
                    pose_frames=pose_frames,
                    validate_pose=False,
                )
                self.tf_mode_queue.popleft()
                continue

            # Look up the transform at the image timestamp
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.tf_world_frame,
                    self.tf_camera_frame,
                    stamp,
                    timeout=rclpy.duration.Duration(seconds=0.0),
                )
            except Exception as e:
                elapsed = time.monotonic() - enqueue_time
                if self.wait_for_first_valid_tf and not self.has_valid_tf:
                    latest_transform = None
                    try:
                        latest_transform = self.tf_buffer.lookup_transform(
                            self.tf_world_frame,
                            self.tf_camera_frame,
                            rclpy.time.Time(),
                            timeout=rclpy.duration.Duration(seconds=0.0),
                        )
                    except Exception:
                        latest_transform = None

                    if latest_transform is not None:
                        transform = latest_transform
                    else:
                        now = time.monotonic()
                        last_warning_time = entry["last_warning_time"]
                        if (
                            elapsed >= self.tf_lookup_timeout
                            and (
                                last_warning_time is None
                                or (now - last_warning_time) >= self.tf_lookup_timeout
                            )
                        ):
                            self.log_frame_sync(
                                status="warming_up",
                                rgb_depth_dt=abs(rgb_timestamp - depth_timestamp),
                                pose_rgb_dt=None,
                                pose_mode=self.pose_represents,
                                pose_frames=pose_frames,
                                reason=f"waiting_for_first_valid_tf({elapsed:.3f}s): {e}",
                            )
                            entry["last_warning_time"] = now
                        break
                elif elapsed < self.tf_lookup_timeout:
                    break  # Break and wait for next timer tick
                else:
                    self.dropped_frame_count += 1
                    self.log_frame_sync(
                        status="dropped",
                        rgb_depth_dt=abs(rgb_timestamp - depth_timestamp),
                        pose_rgb_dt=None,
                        pose_mode=self.pose_represents,
                        pose_frames=pose_frames,
                        reason=f"tf_lookup_timeout({elapsed:.3f}s): {e}",
                    )
                    self.tf_mode_queue.popleft()
                    continue

            # Success: remove from queue
            self.tf_mode_queue.popleft()
            self.mark_valid_tf(pose_frames)

            # Extract translation and quaternion from TransformStamped
            t = transform.transform.translation
            r = transform.transform.rotation
            translation = np.array([t.x, t.y, t.z])
            quaternion = np.array([r.x, r.y, r.z, r.w])
            pose_timestamp = self.stamp_to_seconds(transform.header.stamp)
            if pose_timestamp == 0.0:
                pose_timestamp = rgb_timestamp
            pose_frames = self.format_pose_frames(
                transform.header.frame_id,
                transform.child_frame_id,
                default=pose_frames,
            )

            should_process, _ = self.evaluate_frame_sync(
                rgb_timestamp=rgb_timestamp,
                depth_timestamp=depth_timestamp,
                pose_timestamp=pose_timestamp,
                pose_mode=self.pose_represents,
                pose_frames=pose_frames,
            )
            if not should_process:
                continue

            # Process images (identical to Odometry mode)
            if self.cfg.use_compressed_topic:
                rgb_img = self.decompress_image(rgb_msg.data, is_depth=False)
                depth_img = self.decompress_image(depth_msg.data, is_depth=True)
            else:
                rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
                depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

            depth_factor = getattr(self.dataset_cfg, 'depth_factor', 1000.0)
            depth_img = self.process_depth_image(depth_img, depth_factor)

            timestamp = rgb_timestamp
            pose_matrix = self.build_pose_matrix(translation, quaternion)
            self.push_data(
                rgb_img,
                depth_img,
                pose_matrix,
                timestamp,
                rgb_timestamp=rgb_timestamp,
                depth_timestamp=depth_timestamp,
                pose_timestamp=pose_timestamp,
                pose_mode=self.pose_represents,
            )
            self.last_message_time = self.get_clock().now().nanoseconds / 1e9

    @staticmethod
    def stamp_to_seconds(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

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
    runner = None
    rclpy.init()
    try:
        runner = RunnerROS2(cfg)
        runner.logger.warning("[Main] ROS2 Runner started. Waiting for data stream...")
        while rclpy.ok() and not runner.shutdown_requested:
            rclpy.spin_once(runner, timeout_sec=0.1)
    except KeyboardInterrupt:
        if runner is not None:
            runner.logger.warning("[Main] KeyboardInterrupt received. Shutting down.")
    except ExternalShutdownException:
        if runner is not None:
            runner.logger.warning("[Main] ROS2 context already shutting down.")
    finally:
        if runner is not None:
            runner.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if runner is not None:
            runner.logger.warning("[Main] Done.")
