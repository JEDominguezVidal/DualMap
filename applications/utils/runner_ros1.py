# runner_ros1.py

import logging
import threading
import time
from collections import deque

import numpy as np
import rospy
import tf
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from omegaconf import OmegaConf
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from applications.utils.runner_ros_base import RunnerROSBase
from dualmap.core import Dualmap
from utils.logging_helper import setup_logging


class RunnerROS1(RunnerROSBase):
    """
    ROS1-specific runner, handles topic subscriptions and data flow using rospy.
    """

    def __init__(self, cfg):
        rospy.init_node("runner_ros", anonymous=True)
        setup_logging(output_path=cfg.output_path, config_path=cfg.logging_config)
        self.logger = logging.getLogger(__name__)
        self.logger.info("[Runner ROS1]")
        self.logger.info(OmegaConf.to_yaml(cfg))

        self.cfg = cfg
        self.dualmap = Dualmap(cfg)
        super().__init__(cfg, self.dualmap)

        self.bridge = CvBridge()
        self.dataset_cfg = OmegaConf.load(cfg.ros_stream_config_path)
        self.intrinsics = self.load_intrinsics(self.dataset_cfg)

        # Image and Odometry Subscribers
        if self.cfg.use_compressed_topic:
            self.logger.warning("[Main] Using compressed topics.")
            self.rgb_sub = Subscriber(self.dataset_cfg.ros_topics.rgb, CompressedImage)
            self.depth_sub = Subscriber(
                self.dataset_cfg.ros_topics.depth, CompressedImage
            )
        else:
            self.logger.warning("[Main] Using uncompressed topics.")
            self.rgb_sub = Subscriber(self.dataset_cfg.ros_topics.rgb, Image)
            self.depth_sub = Subscriber(self.dataset_cfg.ros_topics.depth, Image)

        # Detect pose mode: Odometry topic vs TF lookup
        self.use_tf_mode = not hasattr(self.dataset_cfg.ros_topics, "odom")
        self.configure_pose_contract(self.dataset_cfg, self.use_tf_mode)

        if self.use_tf_mode:
            # TF mode: obtain camera pose from tf transforms
            self.logger.warning(
                "[Main] No 'odom' topic defined. Using TF mode for camera pose."
            )
            self.tf_listener = tf.TransformListener()
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
            self.odom_sub = Subscriber(self.dataset_cfg.ros_topics.odom, Odometry)

            # Sync RGB + Depth + Odometry
            self.sync = ApproximateTimeSynchronizer(
                [self.rgb_sub, self.depth_sub, self.odom_sub],
                queue_size=10,
                slop=self.cfg.sync_threshold,
            )
            self.sync.registerCallback(self.synced_callback)

        # Fallback to camera_info topic if intrinsics not loaded
        rospy.Subscriber(
            self.dataset_cfg.ros_topics.camera_info,
            CameraInfo,
            self.camera_info_callback,
        )

    def synced_callback(self, rgb_msg, depth_msg, odom_msg):
        """Callback for synchronised RGB, Depth, and Odom messages (Odometry mode)."""
        rgb_timestamp = rgb_msg.header.stamp.to_sec()
        depth_timestamp = depth_msg.header.stamp.to_sec()
        odom_timestamp = odom_msg.header.stamp.to_sec()
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
        self.last_message_time = time.time()

    def synced_callback_tf(self, rgb_msg, depth_msg):
        """Callback for synchronised RGB-D input using TF for camera pose. Buffers frames."""
        queue_limit = (
            self.tf_runtime_queue_size
            if self.has_valid_tf or not self.wait_for_first_valid_tf
            else self.tf_warmup_queue_size
        )
        if queue_limit > 0 and len(self.tf_mode_queue) >= queue_limit:
            old_entry = self.tf_mode_queue.popleft()
            old_rgb = old_entry["rgb_msg"]
            old_depth = old_entry["depth_msg"]
            rgb_timestamp = old_rgb.header.stamp.to_sec()
            depth_timestamp = old_depth.header.stamp.to_sec()
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
            rgb_timestamp = rgb_msg.header.stamp.to_sec()
            depth_timestamp = depth_msg.header.stamp.to_sec()
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
                (trans, quat) = self.tf_listener.lookupTransform(
                    self.tf_world_frame,
                    self.tf_camera_frame,
                    stamp,
                )
            except Exception as e:
                elapsed = time.monotonic() - enqueue_time
                if self.wait_for_first_valid_tf and not self.has_valid_tf:
                    latest_transform = None
                    try:
                        latest_transform = self.tf_listener.lookupTransform(
                            self.tf_world_frame,
                            self.tf_camera_frame,
                            rospy.Time(0),
                        )
                    except Exception:
                        latest_transform = None

                    if latest_transform is not None:
                        trans, quat = latest_transform
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
                    break
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

            translation = np.array(trans)
            quaternion = np.array(quat)  # ROS1 tf returns (x, y, z, w)
            timestamp = rgb_timestamp
            pose_timestamp = rgb_timestamp

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
            self.last_message_time = time.time()

    def camera_info_callback(self, msg):
        """Fallback callback to get intrinsics from CameraInfo if needed."""
        if self.intrinsics is None:
            self.intrinsics = np.array(msg.K).reshape(3, 3)
            self.logger.warning("[Main] Camera intrinsics received and stored.")

    def spin(self):
        """Main loop calling run_once() at configured ROS rate."""
        rate = rospy.Rate(self.cfg.ros_rate)
        while not rospy.is_shutdown() and not self.shutdown_requested:
            if getattr(self, "use_tf_mode", False):
                self.process_tf_queue()
                
            try:
                self.run_once(lambda: time.time())
            except Exception as e:
                self.logger.error(f"[RunnerROS1] Exception: {e}", exc_info=True)
            rate.sleep()


def run_ros1(cfg):
    """Launch the ROS1 runner in a background thread."""
    runner = RunnerROS1(cfg)
    runner.logger.warning("[Main] ROS1 Runner started. Waiting for data stream...")

    spin_thread = threading.Thread(target=runner.spin)
    spin_thread.start()

    try:
        while not rospy.is_shutdown() and not runner.shutdown_requested:
            time.sleep(0.1)
    except KeyboardInterrupt:
        runner.logger.warning("[Main] KeyboardInterrupt received.")
    finally:
        runner.shutdown_requested = True
        runner.logger.warning("[Main] Shutting down...")
        spin_thread.join(timeout=3.0)

        try:
            rospy.signal_shutdown("User requested shutdown")
        except Exception:
            pass

        runner.logger.warning("[Main] Exit complete.")

        import os

        os._exit(0)
