# runner_ros_base.py

import logging
import time
from collections import deque

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.time_utils import timing_context
from utils.types import DataInput


class RunnerROSBase:
    """
    Base class for ROS1 and ROS2 runners.
    Handles shared logic such as intrinsics/extrinsics loading,
    image decompression, pose conversion, and keyframe processing.
    """

    def __init__(self, cfg, dualmap):
        self.cfg = cfg
        self.dualmap = dualmap
        self.logger = logging.getLogger(__name__)

        self.kf_idx = 0
        self.intrinsics = None
        self.extrinsics = None
        self.pose_represents = None
        self.tf_world_frame = None
        self.tf_camera_frame = None
        self.synced_data_queue = deque(maxlen=1)
        self.shutdown_requested = False
        self.last_message_time = None
        self.max_rgb_depth_dt = float(getattr(cfg, "max_rgb_depth_dt", 0.03))
        self.max_pose_rgb_dt = float(getattr(cfg, "max_pose_rgb_dt", 0.03))
        self.tf_lookup_timeout = float(getattr(cfg, "tf_lookup_timeout", 0.10))
        self.drop_unsynced_frames = bool(getattr(cfg, "drop_unsynced_frames", True))
        self.wait_for_first_valid_tf = bool(
            getattr(cfg, "wait_for_first_valid_tf", True)
        )
        self.tf_runtime_queue_size = int(getattr(cfg, "tf_runtime_queue_size", 50))
        self.tf_warmup_queue_size = int(getattr(cfg, "tf_warmup_queue_size", 512))
        self.dropped_frame_count = 0
        self.has_valid_tf = False

    def load_intrinsics(self, dataset_cfg):
        """Load camera intrinsics from config file."""
        intrinsic_cfg = dataset_cfg.get("intrinsic", None)
        if intrinsic_cfg:
            fx, fy, cx, cy = (
                intrinsic_cfg["fx"],
                intrinsic_cfg["fy"],
                intrinsic_cfg["cx"],
                intrinsic_cfg["cy"],
            )
            self.logger.warning("[Main] Loaded intrinsics from config.")
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.logger.warning("[Main] No intrinsics provided.")
        return None

    def load_extrinsics(self, dataset_cfg):
        """Load camera extrinsics from config file."""
        extrinsic_cfg = dataset_cfg.get("extrinsics", None)
        if extrinsic_cfg is not None:
            matrix = np.array(extrinsic_cfg)
            if matrix.shape == (4, 4):
                self.logger.warning("[Main] Loaded extrinsics from config.")
                return matrix
            raise ValueError(
                "[Main] Invalid extrinsics matrix in config. Expected a 4x4 matrix."
            )
        self.logger.warning("[Main] No extrinsics provided in config.")
        return None

    def resolve_pose_represents(self, dataset_cfg, use_tf_mode):
        """Resolve whether the incoming pose refers to the camera or the base frame."""
        pose_represents = dataset_cfg.get("pose_represents", None)
        if pose_represents is None:
            pose_represents = "camera_frame" if use_tf_mode else "base_frame"
            self.logger.warning(
                "[Main] Missing 'pose_represents' in ROS config. Assuming '%s'.",
                pose_represents,
            )

        valid_modes = {"camera_frame", "base_frame"}
        if pose_represents not in valid_modes:
            raise ValueError(
                "[Main] Invalid pose_represents '%s'. Expected one of %s."
                % (pose_represents, sorted(valid_modes))
            )

        if use_tf_mode and pose_represents != "camera_frame":
            raise ValueError(
                "[Main] TF mode requires pose_represents='camera_frame' because "
                "TF lookup already returns the camera pose."
            )

        return pose_represents

    def configure_pose_contract(self, dataset_cfg, use_tf_mode):
        """Resolve pose semantics and the extrinsics policy used by the runners."""
        self.pose_represents = self.resolve_pose_represents(dataset_cfg, use_tf_mode)
        loaded_extrinsics = self.load_extrinsics(dataset_cfg)
        identity = np.eye(4)

        if self.pose_represents == "camera_frame":
            if loaded_extrinsics is not None and not np.allclose(
                loaded_extrinsics, identity
            ):
                self.logger.warning(
                    "[Main] pose_represents='camera_frame'. Ignoring non-identity "
                    "extrinsics because the incoming pose already describes the camera."
                )
            self.extrinsics = identity
        else:
            if loaded_extrinsics is None:
                raise ValueError(
                    "[Main] pose_represents='base_frame' requires a valid 4x4 "
                    "extrinsics matrix to convert base poses into camera poses."
                )
            self.extrinsics = loaded_extrinsics
            self.logger.warning(
                "[Main] pose_represents='base_frame'. Extrinsics will be applied "
                "to convert base poses into camera poses."
            )

        self.logger.warning(
            "[Main] Pose contract resolved: pose_represents='%s'.",
            self.pose_represents,
        )
        return self.pose_represents

    def resolve_tf_frames(self, dataset_cfg):
        """Resolve canonical TF frame names, accepting legacy aliases with warnings."""
        tf_world_frame = dataset_cfg.get("tf_world_frame", None)
        tf_camera_frame = dataset_cfg.get("tf_camera_frame", None)
        legacy_source = dataset_cfg.get("tf_source_frame", None)
        legacy_target = dataset_cfg.get("tf_target_frame", None)

        if tf_world_frame and tf_camera_frame:
            if legacy_source or legacy_target:
                self.logger.warning(
                    "[Main] Both canonical TF frame keys and deprecated "
                    "tf_source_frame/tf_target_frame were provided. "
                    "Using tf_world_frame/tf_camera_frame."
                )
            return tf_world_frame, tf_camera_frame

        if legacy_source or legacy_target:
            if not (legacy_source and legacy_target):
                raise ValueError(
                    "[Main] Deprecated TF config requires both tf_source_frame and "
                    "tf_target_frame."
                )
            self.logger.warning(
                "[Main] tf_source_frame/tf_target_frame are deprecated. "
                "Please migrate to tf_world_frame/tf_camera_frame."
            )
            return legacy_source, legacy_target

        raise ValueError(
            "[Main] TF mode requires tf_world_frame/tf_camera_frame "
            "(or deprecated tf_source_frame/tf_target_frame)."
        )

    def resolve_camera_pose_matrix(self, pose):
        """Convert the incoming pose into the camera pose expected by the mapper."""
        if self.pose_represents == "camera_frame":
            return pose
        if self.pose_represents == "base_frame":
            if self.extrinsics is None:
                raise RuntimeError(
                    "[Main] Cannot resolve camera pose: extrinsics are missing."
                )
            return pose @ self.extrinsics
        raise RuntimeError("[Main] Pose contract has not been configured yet.")

    def format_pose_frames(self, parent_frame, child_frame, default="unknown"):
        if parent_frame or child_frame:
            return f"{parent_frame or '?'}->{child_frame or '?'}"
        return default

    def mark_valid_tf(self, pose_frames: str) -> None:
        """Mark TF mode as ready after the first successful lookup."""
        if self.has_valid_tf:
            return
        self.has_valid_tf = True
        self.logger.warning(
            "[Main] First valid TF received. TF warm-up finished for %s.",
            pose_frames,
        )

    def log_frame_sync(
        self,
        *,
        status,
        rgb_depth_dt,
        pose_rgb_dt,
        pose_mode,
        pose_frames,
        reason,
    ):
        """Emit a structured per-frame synchronisation log entry."""
        log_method = self.logger.info
        if status != "accepted":
            log_method = self.logger.warning

        pose_rgb_text = (
            f"{pose_rgb_dt:.4f}" if pose_rgb_dt is not None else "nan"
        )
        log_method(
            "[Main][FrameSync] status=%s rgb_depth_dt=%.4f pose_rgb_dt=%s "
            "pose_mode=%s pose_frames=%s reason=%s dropped_count=%d",
            status,
            rgb_depth_dt,
            pose_rgb_text,
            pose_mode,
            pose_frames,
            reason,
            self.dropped_frame_count,
        )

    def evaluate_frame_sync(
        self,
        *,
        rgb_timestamp,
        depth_timestamp,
        pose_timestamp,
        pose_mode,
        pose_frames,
        validate_pose=True,
        log_decision=True,
    ):
        """Validate temporal alignment between RGB, depth, and pose timestamps."""
        rgb_depth_dt = abs(rgb_timestamp - depth_timestamp)
        pose_rgb_dt = None
        reason = "within_thresholds"

        if rgb_depth_dt > self.max_rgb_depth_dt:
            reason = (
                f"rgb_depth_dt_exceeded({rgb_depth_dt:.4f}>{self.max_rgb_depth_dt:.4f})"
            )
        elif validate_pose and pose_timestamp is not None:
            pose_rgb_dt = abs(pose_timestamp - rgb_timestamp)
            if pose_rgb_dt > self.max_pose_rgb_dt:
                reason = (
                    f"pose_rgb_dt_exceeded({pose_rgb_dt:.4f}>{self.max_pose_rgb_dt:.4f})"
                )
        elif pose_timestamp is not None:
            pose_rgb_dt = abs(pose_timestamp - rgb_timestamp)

        within_thresholds = reason == "within_thresholds"
        should_process = within_thresholds or not self.drop_unsynced_frames

        if not within_thresholds and self.drop_unsynced_frames:
            self.dropped_frame_count += 1

        if log_decision:
            status = "accepted"
            if not within_thresholds:
                status = "accepted_with_warning" if should_process else "dropped"
            self.log_frame_sync(
                status=status,
                rgb_depth_dt=rgb_depth_dt,
                pose_rgb_dt=pose_rgb_dt,
                pose_mode=pose_mode,
                pose_frames=pose_frames,
                reason=reason,
            )

        return should_process, {
            "rgb_depth_dt": rgb_depth_dt,
            "pose_rgb_dt": pose_rgb_dt,
            "reason": reason,
        }

    def create_world_transform(self):
        """Create world coordinate transformation from roll/pitch/yaw."""
        roll = np.radians(self.cfg.world_roll)
        pitch = np.radians(self.cfg.world_pitch)
        yaw = np.radians(self.cfg.world_yaw)

        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ]
        )
        Ry = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ]
        )
        Rz = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )

        R_combined = Rz @ Ry @ Rx
        T = np.eye(4)
        T[:3, :3] = R_combined
        return T

    def decompress_image(self, msg_data, is_depth=False):
        """Decode compressed image data (RGB or depth)."""
        msg_data = bytes(msg_data)
        if is_depth:
            depth_data = np.frombuffer(msg_data[12:], np.uint8)
            img = cv2.imdecode(depth_data, cv2.IMREAD_UNCHANGED)
        else:
            np_arr = np.frombuffer(msg_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def process_depth_image(self, depth_img, depth_factor):
        """
        Process depth image to convert to meters (float32) with shape (H, W, 1).

        Supports multiple depth formats:
        - 16UC1 (uint16): Typically in millimeters, requires depth_factor (e.g., 1000.0)
        - 32FC1 (float32): Typically already in meters, use depth_factor=1.0
        - 64FC1 (float64): Typically already in meters, use depth_factor=1.0

        Args:
            depth_img: Raw depth image from sensor/simulator
            depth_factor: Divisor to convert depth values to meters
                         - For uint16 mm depth: use 1000.0
                         - For float32/64 meter depth: use 1.0

        Returns:
            Processed depth image as float32 with shape (H, W, 1)
        """
        if depth_img.dtype == np.uint16:
            # 16UC1: typically millimeters from real depth cameras (RealSense, Orbbec, etc.)
            depth_img = depth_img.astype(np.float32) / depth_factor
        elif depth_img.dtype in [np.float32, np.float64]:
            # 32FC1 or 64FC1: typically meters from simulators (Isaac Sim, Gazebo, etc.)
            depth_img = depth_img.astype(np.float32) / depth_factor
        else:
            self.logger.warning(
                f"[Main] Unexpected depth image dtype: {depth_img.dtype}. "
                "Attempting to convert to float32."
            )
            depth_img = depth_img.astype(np.float32) / depth_factor

        depth_img = np.expand_dims(depth_img, axis=-1)
        return depth_img

    def build_pose_matrix(self, translation, quaternion):
        """Construct 4x4 pose matrix from translation and quaternion."""
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation
        return transformation_matrix

    def push_data(
        self,
        rgb_img,
        depth_img,
        pose,
        timestamp,
        *,
        rgb_timestamp=None,
        depth_timestamp=None,
        pose_timestamp=None,
        pose_mode=None,
    ):
        """Push synchronized input data into queue for processing."""
        camera_pose = self.resolve_camera_pose_matrix(pose)
        transformed_pose = self.create_world_transform() @ camera_pose
        rgb_timestamp = timestamp if rgb_timestamp is None else rgb_timestamp
        depth_timestamp = rgb_timestamp if depth_timestamp is None else depth_timestamp
        pose_timestamp = rgb_timestamp if pose_timestamp is None else pose_timestamp

        data_input = DataInput(
            idx=self.kf_idx,
            time_stamp=timestamp,
            color=rgb_img,
            depth=depth_img,
            color_name=str(timestamp),
            intrinsics=self.intrinsics,
            pose=transformed_pose,
            rgb_timestamp=rgb_timestamp,
            depth_timestamp=depth_timestamp,
            pose_timestamp=pose_timestamp,
            rgb_depth_dt=abs(rgb_timestamp - depth_timestamp),
            pose_rgb_dt=abs(pose_timestamp - rgb_timestamp),
            pose_mode=pose_mode or self.pose_represents or "unknown",
        )
        self.synced_data_queue.append(data_input)
        return data_input

    def mapping_inputs_ready(self):
        """Return whether enough state is available to process a keyframe."""
        if self.intrinsics is None:
            return False, "waiting_for_intrinsics"

        if (
            getattr(self, "use_tf_mode", False)
            and self.wait_for_first_valid_tf
            and not self.has_valid_tf
        ):
            return False, "waiting_for_first_valid_tf"

        return True, "ready"

    def run_once(self, current_time_fn):
        """Check and process a keyframe if data is ready."""
        if not self.synced_data_queue:
            return

        ready, reason = self.mapping_inputs_ready()
        if not ready:
            self.logger.info("[Main] Delaying keyframe processing: %s", reason)
            return

        data_input = self.synced_data_queue[-1]

        if not self.dualmap.calculate_path:
            current_time = current_time_fn()
            last_time = self.last_message_time
            if self.cfg.use_end_process and last_time is not None:
                if current_time - last_time > 20.0:
                    self.logger.warning(
                        "[Main] No new data received. Entering end process."
                    )
                    self.dualmap.end_process()
                    self.shutdown_requested = True
                    return

        if not self.dualmap.check_keyframe(data_input.time_stamp, data_input.pose):
            return

        data_input.idx = self.dualmap.get_keyframe_idx()
        self.logger.info(
            "[Main] Accepted keyframe %d by %s",
            data_input.idx,
            getattr(self.dualmap, "last_keyframe_reason", "unknown"),
        )

        self.logger.info(
            "[Main] ============================================================"
        )
        process_start_time = time.perf_counter()
        with timing_context("Time Per Frame", self.dualmap):
            if self.cfg.use_parallel:
                self.dualmap.parallel_process(data_input)
            else:
                self.dualmap.sequential_process(data_input)

        self.logger.info(
            f"[Main] Processing keyframe {data_input.idx} took {time.perf_counter() - process_start_time:.2f} seconds."
        )
