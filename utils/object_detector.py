import gzip
from collections import Counter
import logging
import os
import pdb
import pickle
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import open_clip
import supervision as sv
import torch
from omegaconf import DictConfig
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import SAM, YOLO, FastSAM

from utils.pcd_utils import (
    mask_depth_to_points,
    refine_points_with_clustering,
    safe_create_bbox,
)
from utils.time_utils import timing_context
from utils.types import DataInput, LocalObservation, ObjectClasses
from utils.visualizer import ReRunVisualizer, visualize_result_rgb

# Set up the module-level logger
logger = logging.getLogger(__name__)


class PoseLowPassFilter:
    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self.initialized = False
        self.smoothed_translation = None
        self.smoothed_rotation = None  # Rotation object (scipy)

    def update(self, pose_mat: np.ndarray) -> np.ndarray:
        """
        input 4x4 pose matrix, output smoothed 4x4 pose matrix.
        """
        curr_translation = pose_mat[:3, 3]
        curr_rotation = R.from_matrix(pose_mat[:3, :3])

        if not self.initialized:
            self.smoothed_translation = curr_translation
            self.smoothed_rotation = curr_rotation
            self.initialized = True
        else:
            # translation filtering
            self.smoothed_translation = (
                self.alpha * self.smoothed_translation
                + (1 - self.alpha) * curr_translation
            )

            # rotation using slerp
            slerp = Slerp(
                [0, 1], R.concatenate([self.smoothed_rotation, curr_rotation])
            )
            self.smoothed_rotation = slerp(1 - self.alpha)

        T_smooth = np.eye(4)
        T_smooth[:3, :3] = self.smoothed_rotation.as_matrix()
        T_smooth[:3, 3] = self.smoothed_translation
        return T_smooth


class Detector:
    # Given input output detection

    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        """
        Initialize the Detector class.

        Parameters:
        cfg (DictConfig): A configuration object containing paths, parameters, and settings for the detector.

        Returns:
        None
        """

        # Object classes
        classes_path = cfg.yolo.classes_path
        if cfg.yolo.use_given_classes:
            classes_path = cfg.yolo.given_classes_path
            logger.info(f"[Detector][Init] Using given classes, path:{classes_path}")

        self.obj_classes = ObjectClasses(
            classes_file_path=classes_path,
            bg_classes=cfg.yolo.bg_classes,
            skip_bg=cfg.yolo.skip_bg,
        )
        self.unknown_class_id = self.obj_classes.get_unknown_class_id()
        self.cfg = cfg
        try:
            self.cfg.unknown_class_id = self.unknown_class_id
        except Exception:
            pass

        # get detection paths
        self.detection_path = Path(cfg.detection_path)
        self.detection_path.mkdir(parents=True, exist_ok=True)

        # Detection results
        # NOTICE: Detection results are stored in Batch, it is not separated by objects
        self.curr_results = {}
        # Data input
        self.curr_data = DataInput()
        self.prev_data = None
        # KF for layout keyframe
        self.prev_kf_data = None

        # masked points and colors
        self.masked_points = []
        self.masked_colors = []
        self.mask_processing_meta = []
        # Observations, a list for each obj observation
        self.curr_observations = []
        self.last_detection_stats = {
            "yolo_base_detections": 0,
            "fastsam_extra_accepted": 0,
            "support_surface_discarded": 0,
            "ambiguous_cluster_discarded": 0,
        }

        # visualizer
        self.visualizer = ReRunVisualizer()
        self.annotated_image = None
        self.annotated_image_support_surface = None

        # Variables for FastSAM
        if self.unknown_class_id is None and (
            cfg.use_fastsam or cfg.clip_unknown_relabel.enabled
        ):
            raise ValueError(
                "The active class list must contain an 'unknown' label when FastSAM "
                "or CLIP unknown relabeling is enabled."
            )
        self.annotated_image_fs = None
        self.annotated_image_fs_after = None
        self.annotated_image_clip_relabel = None
        self.fastsam_detections = {}
        self.relabel_candidate_ids = np.empty(0, dtype=np.int64)

        # Layout Pointcloud
        self.layout_pointcloud = o3d.geometry.PointCloud()
        self.layout_num = 0
        self.layout_time = 0.0
        # For thread processing
        self.layout_lock = (
            threading.Lock()
        )  # Thread lock for protecting layout_pointcloud
        self.data_thread = None  # Thread handle
        self.data_event = threading.Event()  # Thread notification event

        logger.info(f"[Detector][Init] Initilizating detection modules...")

        if cfg.run_detection:
            try:
                # CLIP module
                logger.info(
                    f"[Detector][Init] Loading CLIP model: {cfg.clip.model_name} with pretrained weights '{cfg.clip.pretrained}'"
                )

                # MobileCLIP2 S0/S2/B models need custom image normalization
                model_kwargs = {}
                model_name = cfg.clip.model_name
                if model_name.startswith("MobileCLIP2") and not (
                    model_name.endswith("S3")
                    or model_name.endswith("S4")
                    or model_name.endswith("L-14")
                ):
                    model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

                self.clip_model, _, self.clip_preprocess = (
                    open_clip.create_model_and_transforms(
                        cfg.clip.model_name,
                        pretrained=cfg.clip.pretrained,
                        **model_kwargs,
                    )
                )
                self.clip_model = self.clip_model.to(cfg.device)
                self.clip_model.eval()

                # Only reparameterize if the model is MobileCLIP
                if "MobileCLIP" in cfg.clip.model_name:
                    from mobileclip.modules.common.mobileone import reparameterize_model

                    self.clip_model = reparameterize_model(self.clip_model)

                self.clip_tokenizer = open_clip.get_tokenizer(cfg.clip.model_name)
            except Exception as e:
                logger.error(f"[Detector][Init] Error loading CLIP model: {e}")
                return

            try:
                # Detection module
                logger.info(
                    f"[Detector][Init] Loading YOLO model from\t{cfg.yolo.model_path}"
                )
                self.yolo = YOLO(cfg.yolo.model_path)
                self.yolo.set_classes(self.obj_classes.get_classes_arr())
            except Exception as e:
                logger.error(f"[Detector][Init] Error loading YOLO model: {e}")
                return

            try:
                # Segmentation module
                logger.info(
                    f"[Detector][Init] Loading SAM model from\t{cfg.sam.model_path}"
                )
                self.sam = SAM(cfg.sam.model_path)
            except Exception as e:
                logger.error(f"[Detector][Init] Error loading SAM model: {e}")
                return

            # Open fastsam for open vocabulary detection
            if cfg.use_fastsam:
                try:
                    logger.info(
                        f"[Detector][Init] Loading FastSAM model from\t{cfg.fastsam.model_path}"
                    )
                    self.fastsam = FastSAM(cfg.fastsam.model_path)
                except Exception as e:
                    logger.error(f"[Detector][Init] Error loading FASTSAM model: {e}")
                    return

            logger.info("[Detector][Init] Initializing high-low mobility classifier.")
            lm_examples = cfg.lm_examples
            hm_examples = cfg.hm_examples
            lm_descriptions = cfg.lm_descriptions
            num_examples = [len(lm_examples), len(hm_examples), len(lm_descriptions)]
            prototypes = lm_examples + hm_examples + lm_descriptions
            proto_feats = get_text_features(
                prototypes,
                self.clip_model,
                self.clip_tokenizer,
                device=cfg.device,
                clip_length=cfg.clip.clip_length,
            )
            self.num_examples = num_examples
            self.proto_feats = proto_feats

            # Get the text feats of all the classes
            class_feats = get_text_features(
                self.obj_classes.get_classes_arr(),
                self.clip_model,
                self.clip_tokenizer,
                device=cfg.device,
                clip_length=cfg.clip.clip_length,
            )
            self.class_feats = class_feats
            self.relabel_candidate_ids = self.build_relabel_candidate_ids()

            if cfg.clip_unknown_relabel.enabled and len(self.relabel_candidate_ids) == 0:
                raise ValueError(
                    "CLIP unknown relabeling is enabled, but no candidate classes "
                    "remain after applying the configured exclusions."
                )

            # Used for unknown class
            if cfg.use_avg_feat_for_unknown:
                class_feats_mean = np.mean(class_feats, axis=0)
                self.class_feats_mean = class_feats_mean / np.linalg.norm(
                    class_feats_mean
                )

            with timing_context("Detection Filter", self):
                self.filter = Filter(
                    classes=self.obj_classes,
                    small_mask_size=self.cfg.small_mask_th,
                    skip_refinement=self.cfg.skip_refinement,
                )
                self.filter.set_device(self.cfg.device)

        # for filtering the pose of follower camera for visualization
        self.pose_filter_follower = PoseLowPassFilter(alpha=0.95)

        logger.info(f"[Detector][Init] Finish Init.")

    def update_state(self) -> None:
        self.curr_results = {}
        self.curr_observations = []
        self.mask_processing_meta = []
        self.last_detection_stats = {
            "yolo_base_detections": 0,
            "fastsam_extra_accepted": 0,
            "support_surface_discarded": 0,
            "ambiguous_cluster_discarded": 0,
        }
        # Keep the latest debug images alive until ROS publishes them.
        # self.prev_data = self.curr_data.copy()
        # self.curr_data.clear()

    def update_data(self) -> None:
        # self.curr_results = {}
        # self.curr_observations = []
        self.prev_data = self.curr_data.copy()

        # self.curr_data.clear()

    def set_data_input(self, curr_data: DataInput) -> None:
        self.curr_data = curr_data

        if not self.cfg.preload_layout:
            # If a thread is already running, wait for it to finish
            if self.data_thread and self.data_thread.is_alive():
                self.data_thread.join()

            # Create a new thread to process data input
            self.data_thread = threading.Thread(target=self._process_data_input_thread)
            self.data_thread.start()

    def _process_data_input_thread(self):
        """
        Logic executed in the background thread.
        """
        # Initialize prev_kf_data and layout_pointcloud
        if self.prev_kf_data is None:
            self.prev_kf_data = self.curr_data.copy()
            layout_pcd = self.depth_to_point_cloud(sample_rate=16)
            with self.layout_lock:  # Ensure thread safety for layout_pointcloud
                self.layout_pointcloud += layout_pcd.voxel_down_sample(
                    voxel_size=self.cfg.layout_voxel_size
                )
            logger.info(
                f"[Detector][Layout] Initialized layout pointcloud with {len(self.layout_pointcloud.points)} points."
            )
            return

        # Print current frame index
        logger.info(f"[Detector][Layout] Processing frame idx: {self.curr_data.idx}")

        # Check if layout_pointcloud needs to be updated
        if self.check_keyframe_for_layout_pcd():
            start_time = time.time()

            # Generate current frame point cloud
            current_pcd = self.depth_to_point_cloud(sample_rate=16)

            # Merge point clouds
            with self.layout_lock:
                self.layout_pointcloud += current_pcd
                logger.info(
                    f"[Detector][Layout] Points before downsample: {len(self.layout_pointcloud.points)}"
                )
                self.layout_pointcloud = self.layout_pointcloud.voxel_down_sample(
                    voxel_size=self.cfg.layout_voxel_size
                )
                logger.info(
                    f"[Detector][Layout] Points after downsample: {len(self.layout_pointcloud.points)}"
                )

            # Update prev_kf_data
            self.prev_kf_data = self.curr_data.copy()
            logger.info("[Detector][Layout] Updated layout pointcloud.")

            # Update time and count
            end_time = time.time()
            layout_time = end_time - start_time
            self.layout_time += layout_time
            self.layout_num += 1
            logger.info(
                f"[Detector][Layout] Layout update took {layout_time:.4f} seconds."
            )

    def get_layout_pointcloud(self):
        """
        Return the current layout_pointcloud.
        """
        with self.layout_lock:
            return self.layout_pointcloud

    def save_layout(self):
        if self.layout_pointcloud is not None:
            layout_pcd = self.get_layout_pointcloud()
            save_dir = self.cfg.map_save_path
            layout_pcd_path = save_dir + "/layout.pcd"
            o3d.io.write_point_cloud(layout_pcd_path, layout_pcd)
            logger.info(f"[Detector][Layout] Saving layout to: {layout_pcd_path}")

    def load_layout(self):
        """
        Load layout point cloud layout.pcd, prefer preload_path,
        if not exist, use map_save_path. Skip loading if path or file is missing.
        """
        # Prefer preload_layout_path, if not exist then use map_save_path
        if os.path.exists(self.cfg.preload_path):
            load_dir = self.cfg.preload_path
            logger.info(f"[Detector][Layout] Using preload layout path: {load_dir}")
        else:
            load_dir = self.cfg.map_save_path
            logger.info(
                f"[Detector][Layout] Preload layout path not found. Using default map save path: {load_dir}"
            )

        # Build layout point cloud file path
        layout_pcd_path = os.path.join(load_dir, "layout.pcd")

        # Check if layout point cloud file exists
        if not Path(layout_pcd_path).is_file():
            logger.info(
                f"[Detector][Layout] Layout file not found at: {layout_pcd_path}"
            )
            return None

        # Load layout point cloud
        layout_pcd = o3d.io.read_point_cloud(layout_pcd_path)
        logger.info(f"[Detector][Layout] Layout loaded from: {layout_pcd_path}")

        # Save to class attribute
        self.layout_pointcloud = layout_pcd

    def get_curr_data(
        self,
    ) -> DataInput:
        return self.curr_data

    def get_curr_observations(self) -> None:
        return self.curr_observations

    def check_keyframe_for_layout_pcd(self):
        """
        Check if the current frame should be selected as a keyframe based on
        time interval, pose difference (translation), and rotation difference.
        """
        curr_pose = self.curr_data.pose
        prev_kf_pose = self.prev_kf_data.pose

        # Translation check
        translation_diff = np.linalg.norm(
            curr_pose[:3, 3] - prev_kf_pose[:3, 3]
        )  # Translation difference
        if translation_diff >= self.cfg.layout_translation_threshold:
            logger.info(
                f"[Detector][Layout] Candidate Frame for layout calculation -- translation: {translation_diff}"
            )
            return True

        # Rotation check
        curr_rotation = R.from_matrix(curr_pose[:3, :3])
        last_rotation = R.from_matrix(prev_kf_pose[:3, :3])
        rotation_diff = curr_rotation.inv() * last_rotation
        angle_diff = rotation_diff.magnitude() * (180 / np.pi)

        if angle_diff >= self.cfg.layout_rotation_threshold:
            logger.info(
                f"[Detector][Layout] Candidate Frame for layout calculation -- rotation: {angle_diff}"
            )
            return True

        return False

    def process_yolo_results(self, color, obj_classes):

        # Perform YOLO prediction
        results = self.yolo.predict(color, conf=0.2, verbose=False)

        # Extract confidence scores
        confidence_tensor = results[0].boxes.conf
        confidence_np = confidence_tensor.cpu().numpy()

        # Extract class IDs
        detection_class_id_tensor = results[0].boxes.cls
        detection_class_id_np = detection_class_id_tensor.cpu().numpy().astype(int)

        # Generate class labels
        detection_class_labels = [
            f"{obj_classes.get_classes_arr()[class_id]} {class_idx}"
            for class_idx, class_id in enumerate(detection_class_id_np)
        ]

        # Extract bounding box coordinates
        xyxy_tensor = results[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()

        return confidence_np, detection_class_id_np, detection_class_labels, xyxy_np

    def process_fastsam_results(self, color):
        results = self.fastsam(
            color,
            device="cuda",
            retina_masks=True,
            imgsz=1024,
            conf=self.cfg.fastsam_confidence,
            iou=0.9,
            verbose=False,
        )
        # Extract confidence scores
        confidence_tensor = results[0].boxes.conf
        confidence_np = confidence_tensor.cpu().numpy()

        # Extract bounding box coordinates
        xyxy_tensor = results[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()

        # Extract Masks with protection against None
        if results[0].masks is not None:
            masks_tensor = results[0].masks.data
            masks_np = masks_tensor.cpu().numpy().astype(bool)
        else:
            logging.warning(
                "[Detector] fastSAM did not return any masks, using empty mask array"
            )
            # If no mask is returned, create an empty array. Assume mask size matches input image's first two dims
            masks_np = np.empty((0,) + color.shape[:2], dtype=bool)

        # Extract class IDs (default all set to unknown_class_id)
        detection_class_id_tensor = results[0].boxes.cls
        detection_class_id_np = detection_class_id_tensor.cpu().numpy().astype(int)
        detection_class_id_np = np.full_like(
            detection_class_id_np, self.unknown_class_id
        )

        return confidence_np, detection_class_id_np, xyxy_np, masks_np

    def merge_detections(self, detections1, detections2):
        # Check if first detections is empty
        if len(detections1.xyxy) == 0:
            return detections2

        # Check if second detections is empty
        if len(detections2.xyxy) == 0:
            return detections1

        # Merge xyxy
        merged_xyxy = np.concatenate([detections1.xyxy, detections2.xyxy], axis=0)

        # Merge confidence
        merged_confidence = np.concatenate(
            [detections1.confidence, detections2.confidence], axis=0
        )

        # Merge class_id
        merged_class_id = np.concatenate(
            [detections1.class_id, detections2.class_id], axis=0
        )

        # Merge mask
        merged_masks = np.concatenate([detections1.mask, detections2.mask], axis=0)

        # Create new sv.Detections object
        merged_detections = sv.Detections(
            xyxy=merged_xyxy,
            confidence=merged_confidence,
            class_id=merged_class_id,
            mask=merged_masks,
        )

        return merged_detections

    def process_fastsam(self, color):

        with timing_context("FastSAM", self):
            fs_confidence_np, fs_class_id_np, fs_xyxy_np, fs_masks_np = (
                self.process_fastsam_results(color)
            )

        if len(fs_confidence_np) == 0:
            logger.warning("[Detector] No detections found in curr frame by FastSAM.")
            self.fastsam_detections = {}
            return

        # debug fastsam
        fs_detections = sv.Detections(
            xyxy=fs_xyxy_np,
            confidence=fs_confidence_np,
            class_id=fs_class_id_np,
            mask=fs_masks_np,
        )

        if self.cfg.visualize_detection and self.cfg.show_fastsam_debug:
            image_fs, _ = visualize_result_rgb(
                color, fs_detections, self.obj_classes.get_classes_arr()
            )

            self.annotated_image_fs = image_fs

        self.fastsam_detections = fs_detections

    def process_yolo_and_sam(self, color):
        with timing_context("YOLO", self):
            confidence, class_id, class_labels, xyxy = self.process_yolo_results(
                color, self.obj_classes
            )

        # if detection is empty, return
        if len(confidence) == 0:
            logger.warning("[Detector] No detections found in curr frame.")
            self.curr_detections = sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=np.int64),
                mask=np.empty((0,) + color.shape[:2], dtype=bool),
            )
            return
        with timing_context("Segmentation", self):
            sam_out = self.sam.predict(color, bboxes=xyxy, verbose=False)
            masks_tensor = sam_out[0].masks.data
            masks_np = masks_tensor.cpu().numpy()
            self.masks_np = masks_np

        curr_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            mask=masks_np,
        )

        self.curr_detections = curr_detections

    def filter_fs_detections_by_curr(
        self,
        fs_detections,
        curr_detections,
        iou_threshold=0.5,
        overlap_threshold=0.6,
        coarse_mask_ratio=2.5,
    ):
        fastsam_mode = self.get_fastsam_mode()
        if fastsam_mode == "off":
            return self.slice_detections(
                fs_detections,
                np.zeros(len(fs_detections.xyxy), dtype=bool),
            )

        if fastsam_mode == "uncovered_only":
            return self.filter_fs_detections_uncovered_only(
                fs_detections,
                curr_detections,
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert numpy arrays to torch tensors and move to GPU
        fs_masks = torch.tensor(
            fs_detections.mask, dtype=torch.bool, device=device
        )  # (N1, H, W)
        fs_xyxy = torch.tensor(
            fs_detections.xyxy, dtype=torch.float32, device=device
        )  # (N1, 4)
        fs_confidence = torch.tensor(
            fs_detections.confidence, dtype=torch.float32, device=device
        )
        fs_class_id = torch.tensor(
            fs_detections.class_id, dtype=torch.int64, device=device
        )

        curr_masks = torch.tensor(
            curr_detections.mask, dtype=torch.bool, device=device
        )  # (N2, H, W)

        # Get total number of pixels in masks
        num_fs = fs_masks.shape[0]  # N1
        num_curr = curr_masks.shape[0]  # N2

        # Flatten masks to (N, H * W) and convert to float32 for matrix multiplication
        fs_masks_flat = fs_masks.view(num_fs, -1).to(torch.float32)  # (N1, H * W)
        curr_masks_flat = curr_masks.view(num_curr, -1).to(torch.float32)  # (N2, H * W)

        # Compute intersection and union (using float operations)
        intersection = torch.matmul(fs_masks_flat, curr_masks_flat.T)  # (N1, N2)
        fs_area = fs_masks_flat.sum(dim=1, keepdim=True)  # (N1, 1)
        curr_area = curr_masks_flat.sum(dim=1).unsqueeze(0)  # (1, N2)
        union = fs_area + curr_area - intersection  # (N1, N2)

        # Compute IoU
        iou_matrix = intersection / torch.clamp(union, min=1e-7)  # (N1, N2)

        # Compute overlap ratio
        overlap_ratio_fs = intersection / torch.clamp(fs_area, min=1e-7)  # (N1, N2)
        overlap_ratio_curr = intersection / torch.clamp(curr_area, min=1e-7)  # (N1, N2)

        # Initialize keep mask, default is to keep all fs_masks
        keep_mask = torch.ones(num_fs, dtype=torch.bool, device=device)

        fs_box_area = (fs_xyxy[:, 2] - fs_xyxy[:, 0]) * (fs_xyxy[:, 3] - fs_xyxy[:, 1])
        curr_box_area = (curr_detections.xyxy[:, 2] - curr_detections.xyxy[:, 0]) * (
            curr_detections.xyxy[:, 3] - curr_detections.xyxy[:, 1]
        )
        curr_box_area = torch.tensor(curr_box_area, dtype=torch.float32, device=device)

        # Filter masks one by one
        for i in range(num_fs):
            # Check if current fs_mask overlaps with curr_mask
            overlap = (
                (iou_matrix[i] > iou_threshold)
                | (overlap_ratio_fs[i] > overlap_threshold)
                | (overlap_ratio_curr[i] > overlap_threshold)
            )

            if not overlap.any():
                continue

            overlap_indices = torch.where(overlap)[0]
            coarse_overlap = False
            for curr_idx in overlap_indices.tolist():
                area_ratio = curr_box_area[curr_idx] / torch.clamp(fs_box_area[i], min=1.0)
                if area_ratio > coarse_mask_ratio:
                    coarse_overlap = True
                    break

            # Keep the finer FastSAM mask when the overlapping YOLO/SAM mask
            # is much coarser. The final geometric filter will decide whether
            # the mask corresponds to a valid object or a support surface.
            if not coarse_overlap:
                keep_mask[i] = False

        # Filter detections based on keep mask
        filtered_fs_detections = sv.Detections(
            xyxy=fs_xyxy[keep_mask].cpu().numpy(),
            confidence=fs_confidence[keep_mask].cpu().numpy(),
            class_id=fs_class_id[keep_mask].cpu().numpy(),
            mask=fs_masks[keep_mask].cpu().numpy(),
        )

        return filtered_fs_detections

    def is_rosbag_tf_mode(self) -> bool:
        return getattr(self.cfg, "dataset_name", "") == "rosbag_tf"

    def get_fastsam_mode(self) -> str:
        configured_mode = getattr(self.cfg, "fastsam_mode", None)
        if configured_mode is not None:
            return str(configured_mode)
        return "uncovered_only" if self.is_rosbag_tf_mode() else "full"

    def should_clip_relabel_fastsam(self) -> bool:
        configured_value = getattr(self.cfg, "clip_relabel_fastsam", None)
        if configured_value is not None:
            return bool(configured_value)
        return not self.is_rosbag_tf_mode()

    def filter_fs_detections_uncovered_only(self, fs_detections, curr_detections):
        num_fs = len(fs_detections.xyxy)
        if num_fs == 0:
            return fs_detections

        if curr_detections.mask is not None and len(curr_detections.mask) > 0:
            covered_mask = np.any(curr_detections.mask, axis=0)
            image_shape = curr_detections.mask.shape[1:]
        else:
            image_shape = fs_detections.mask.shape[1:]
            covered_mask = np.zeros(image_shape, dtype=bool)

        max_area_ratio = float(
            getattr(self.cfg, "fastsam_uncovered_max_area_ratio", 0.12)
        )
        min_mask_pixels = max(
            int(getattr(self.cfg, "fastsam_uncovered_min_mask_pixels", 80)),
            int(getattr(self.cfg, "small_mask_th", 20)),
            20,
        )
        min_novel_pixels = max(
            int(getattr(self.cfg, "fastsam_uncovered_min_novel_pixels", 150)),
            min_mask_pixels,
        )
        min_novel_ratio = float(
            getattr(self.cfg, "fastsam_uncovered_min_novel_ratio", 0.55)
        )
        max_masks = int(getattr(self.cfg, "fastsam_uncovered_max_masks", 8))

        candidates = []
        for det_idx in range(num_fs):
            mask = np.asarray(fs_detections.mask[det_idx], dtype=bool)
            mask_pixels = int(np.sum(mask))
            if mask_pixels < min_mask_pixels:
                continue

            mask_area_ratio = mask_pixels / max(float(mask.size), 1.0)
            if mask_area_ratio > max_area_ratio:
                continue

            novel_mask = np.logical_and(mask, np.logical_not(covered_mask))
            novel_pixels = int(np.sum(novel_mask))
            novel_ratio = novel_pixels / max(mask_pixels, 1)

            if novel_pixels < min_novel_pixels or novel_ratio < min_novel_ratio:
                continue

            candidates.append((novel_pixels, novel_ratio, mask_pixels, det_idx, mask))

        if not candidates:
            return self.slice_detections(
                fs_detections,
                np.zeros(num_fs, dtype=bool),
            )

        keep_mask = np.zeros(num_fs, dtype=bool)
        kept_count = 0
        for _, _, _, det_idx, mask in sorted(
            candidates,
            key=lambda item: (item[0], item[1], item[2]),
            reverse=True,
        ):
            novel_mask = np.logical_and(mask, np.logical_not(covered_mask))
            novel_pixels = int(np.sum(novel_mask))
            novel_ratio = novel_pixels / max(int(np.sum(mask)), 1)

            if novel_pixels < min_novel_pixels or novel_ratio < min_novel_ratio:
                continue

            keep_mask[det_idx] = True
            covered_mask = np.logical_or(covered_mask, mask)
            kept_count += 1
            if max_masks > 0 and kept_count >= max_masks:
                break

        return self.slice_detections(fs_detections, keep_mask)

    def add_extra_detections_from_fastsam(
        self, color, fastsam_detections, incoming_detections
    ):

        with timing_context("mask_filter", self):
            fs_after_detections = self.filter_fs_detections_by_curr(
                fastsam_detections, incoming_detections
            )

        if self.cfg.visualize_detection and self.cfg.show_fastsam_debug:
            image_fs_after, _ = visualize_result_rgb(
                color, fs_after_detections, self.obj_classes.get_classes_arr()
            )
            self.annotated_image_fs_after = image_fs_after

        accepted_count = len(fs_after_detections.xyxy)

        # merge_detctions
        merged_detctions = self.merge_detections(
            fs_after_detections, incoming_detections
        )
        return merged_detctions, accepted_count

    def slice_detections(self, detections: sv.Detections, keep_mask: np.ndarray):
        """Return a sliced detections object using a boolean keep mask."""
        keep_mask = np.asarray(keep_mask, dtype=bool)
        return sv.Detections(
            xyxy=np.array(detections.xyxy[keep_mask], dtype=np.float32),
            confidence=np.array(detections.confidence[keep_mask], dtype=np.float32),
            class_id=np.array(detections.class_id[keep_mask], dtype=np.int64),
            mask=np.array(detections.mask[keep_mask], dtype=np.bool_),
        )

    def estimate_support_plane(self):
        """Estimate the dominant plane in the current frame for support filtering."""
        filter_cfg = getattr(self.cfg, "support_surface_filter", None)
        if not filter_cfg or not getattr(filter_cfg, "enabled", False):
            return None

        frame_pcd = self.depth_to_point_cloud(sample_rate=16)
        if len(frame_pcd.points) < 64:
            return None

        try:
            plane_model, inliers = frame_pcd.segment_plane(
                distance_threshold=filter_cfg.max_plane_distance,
                ransac_n=3,
                num_iterations=100,
            )
        except RuntimeError:
            return None

        if len(inliers) < 64:
            return None

        return np.asarray(plane_model, dtype=np.float32)

    def build_support_surface_debug_image(
        self,
        color: np.ndarray,
        detections: sv.Detections,
        non_trackable_mask: np.ndarray,
    ) -> np.ndarray:
        support_indices = np.flatnonzero(non_trackable_mask)
        if len(support_indices) == 0:
            return color.copy()

        support_detections = self.slice_detections(detections, non_trackable_mask)
        annotated_image, _ = visualize_result_rgb(
            color,
            support_detections,
            self.obj_classes.get_classes_arr(),
        )
        return annotated_image

    def apply_support_surface_filter(
        self,
        *,
        color: np.ndarray,
        detections: sv.Detections,
        image_feats: np.ndarray,
        text_feats: np.ndarray,
        semantic_confidence: np.ndarray,
        label_source: np.ndarray,
        relabel_top1_scores: np.ndarray,
        relabel_top2_scores: np.ndarray,
        relabel_top1_class_ids: np.ndarray,
        relabel_top2_class_ids: np.ndarray,
    ):
        filter_cfg = getattr(self.cfg, "support_surface_filter", None)
        num_detections = len(detections.xyxy)
        non_trackable_mask = np.zeros(num_detections, dtype=bool)

        if not filter_cfg or not getattr(filter_cfg, "enabled", False) or num_detections == 0:
            self.annotated_image_support_surface = None
            return (
                detections,
                image_feats,
                text_feats,
                semantic_confidence,
                label_source,
                relabel_top1_scores,
                relabel_top2_scores,
                relabel_top1_class_ids,
                relabel_top2_class_ids,
                non_trackable_mask,
            )

        plane_model = self.estimate_support_plane()
        image_area = float(color.shape[0] * color.shape[1])

        for det_idx in range(num_detections):
            if det_idx >= len(self.masked_points) or self.masked_points[det_idx] is None:
                continue

            points = self.masked_points[det_idx]
            if len(points) == 0:
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.transform(self.curr_data.pose)
            bbox = safe_create_bbox(pcd)
            extent = bbox.get_extent()
            mask_area_ratio = float(np.sum(detections.mask[det_idx])) / max(image_area, 1.0)

            flat_large_support = (
                mask_area_ratio > filter_cfg.max_image_area_ratio
                and extent[2] < filter_cfg.max_z_extent
                and max(extent[0], extent[1]) > filter_cfg.min_xy_extent
            )

            plane_ratio = 0.0
            if plane_model is not None:
                plane_normal = plane_model[:3]
                plane_norm = np.linalg.norm(plane_normal)
                if plane_norm > 1e-8:
                    world_points = np.asarray(pcd.points)
                    distances = np.abs(world_points @ plane_normal + plane_model[3]) / plane_norm
                    plane_ratio = float(
                        np.mean(distances <= filter_cfg.max_plane_distance)
                    )

            support_surface = flat_large_support or (
                plane_ratio >= filter_cfg.min_plane_ratio
                and mask_area_ratio > filter_cfg.max_image_area_ratio
            )
            if support_surface:
                non_trackable_mask[det_idx] = True

        self.annotated_image_support_surface = self.build_support_surface_debug_image(
            color=color,
            detections=detections,
            non_trackable_mask=non_trackable_mask,
        )
        self.last_detection_stats["support_surface_discarded"] = int(
            np.sum(non_trackable_mask)
        )

        return (
            detections,
            image_feats,
            text_feats,
            semantic_confidence,
            label_source,
            relabel_top1_scores,
            relabel_top2_scores,
            relabel_top1_class_ids,
            relabel_top2_class_ids,
            non_trackable_mask,
        )

    def build_relabel_candidate_ids(self):
        classes = self.obj_classes.get_classes_arr()
        seen_names = set()
        duplicate_names = set()
        candidate_ids = []

        for idx, class_name in enumerate(classes):
            if class_name in seen_names:
                duplicate_names.add(class_name)
                continue

            seen_names.add(class_name)

            if (
                self.cfg.clip_unknown_relabel.exclude_unknown_label
                and class_name == "unknown"
            ):
                continue

            if (
                self.cfg.clip_unknown_relabel.exclude_bg_classes
                and class_name in self.obj_classes.get_bg_classes_arr()
            ):
                continue

            candidate_ids.append(idx)

        if duplicate_names:
            duplicate_names = sorted(duplicate_names)
            logger.warning(
                "[Detector][Init] Duplicate class names detected in %s. "
                "CLIP relabeling will deduplicate them by keeping the first "
                "occurrence: %s",
                self.obj_classes.classes_file_path,
                duplicate_names,
            )

        return np.asarray(candidate_ids, dtype=np.int64)

    @staticmethod
    def normalize_features(features: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return features / norms

    def relabel_unknown_with_clip(
        self,
        class_ids: np.ndarray,
        confidences: np.ndarray,
        image_feats: np.ndarray,
        text_feats: np.ndarray,
    ):
        relabel_cfg = self.cfg.clip_unknown_relabel
        num_detections = len(class_ids)

        semantic_confidence = confidences.astype(np.float32, copy=True)
        label_source = np.full(num_detections, "yolo", dtype="<U32")
        top1_scores = np.full(num_detections, np.nan, dtype=np.float32)
        top2_scores = np.full(num_detections, np.nan, dtype=np.float32)
        top1_class_ids = np.full(num_detections, -1, dtype=np.int64)
        top2_class_ids = np.full(num_detections, -1, dtype=np.int64)

        if self.unknown_class_id is not None:
            label_source[class_ids == self.unknown_class_id] = "fastsam_unknown"

        if (
            not relabel_cfg.enabled
            or not self.should_clip_relabel_fastsam()
            or self.unknown_class_id is None
            or len(self.relabel_candidate_ids) == 0
        ):
            return (
                class_ids,
                text_feats,
                semantic_confidence,
                label_source,
                top1_scores,
                top2_scores,
                top1_class_ids,
                top2_class_ids,
            )

        unknown_indices = np.where(class_ids == self.unknown_class_id)[0]
        if len(unknown_indices) == 0:
            return (
                class_ids,
                text_feats,
                semantic_confidence,
                label_source,
                top1_scores,
                top2_scores,
                top1_class_ids,
                top2_class_ids,
            )

        if relabel_cfg.use_image_features_only:
            relabel_feats = image_feats[unknown_indices]
        else:
            weighted_feats = (
                self.cfg.image_weight * image_feats[unknown_indices]
                + (1 - self.cfg.image_weight) * text_feats[unknown_indices]
            )
            relabel_feats = self.normalize_features(weighted_feats)

        candidate_feats = self.class_feats[self.relabel_candidate_ids]
        relabel_feats = self.normalize_features(relabel_feats)
        candidate_feats = self.normalize_features(candidate_feats)
        similarity = np.matmul(relabel_feats, candidate_feats.T)

        for row_idx, det_idx in enumerate(unknown_indices):
            scores = similarity[row_idx]
            rank = np.argsort(scores)[::-1]
            top1_rank = rank[0]
            top1_class_id = int(self.relabel_candidate_ids[top1_rank])
            top1_score = float(scores[top1_rank])

            top2_score = -1.0
            top2_class_id = -1
            if len(rank) > 1:
                top2_rank = rank[1]
                top2_class_id = int(self.relabel_candidate_ids[top2_rank])
                top2_score = float(scores[top2_rank])

            top1_scores[det_idx] = top1_score
            top2_scores[det_idx] = top2_score
            top1_class_ids[det_idx] = top1_class_id
            top2_class_ids[det_idx] = top2_class_id

            if (
                top1_score >= relabel_cfg.min_similarity
                and top1_score - top2_score >= relabel_cfg.min_margin
            ):
                class_ids[det_idx] = top1_class_id
                text_feats[det_idx] = self.class_feats[top1_class_id]
                semantic_confidence[det_idx] = top1_score
                label_source[det_idx] = "fastsam_clip"

        relabeled_count = np.sum(label_source == "fastsam_clip")
        if relabeled_count > 0:
            logger.info(
                "[Detector] Relabeled %d unknown detections with CLIP.",
                int(relabeled_count),
            )

        return (
            class_ids,
            text_feats,
            semantic_confidence,
            label_source,
            top1_scores,
            top2_scores,
            top1_class_ids,
            top2_class_ids,
        )

    def log_relabel_summary(
        self,
        label_source: np.ndarray,
        num_detections: int,
    ) -> None:
        if not self.should_clip_relabel_fastsam():
            return
        label_source_counter = Counter(label_source.tolist())
        logger.info(
            "[Detector][Relabel] detections=%d, fastsam_unknown=%d, fastsam_clip=%d, sources=%s",
            num_detections,
            int(label_source_counter.get("fastsam_unknown", 0)),
            int(label_source_counter.get("fastsam_clip", 0)),
            dict(sorted(label_source_counter.items())),
        )

    def build_clip_relabel_debug_image(
        self,
        color: np.ndarray,
        detections: sv.Detections,
        label_source: np.ndarray,
        semantic_confidence: np.ndarray,
    ) -> np.ndarray:
        clip_indices = np.flatnonzero(label_source == "fastsam_clip")
        if len(clip_indices) == 0:
            return color.copy()

        clip_relabel_detections = sv.Detections(
            xyxy=detections.xyxy[clip_indices],
            mask=(
                detections.mask[clip_indices]
                if detections.mask is not None
                else None
            ),
            confidence=semantic_confidence[clip_indices],
            class_id=detections.class_id[clip_indices],
        )

        annotated_image, _ = visualize_result_rgb(
            color,
            clip_relabel_detections,
            self.obj_classes.get_classes_arr(),
        )
        return annotated_image

    def process_detections(self):

        color = self.curr_data.color.astype(np.uint8)
        self.last_detection_stats = {
            "yolo_base_detections": 0,
            "fastsam_extra_accepted": 0,
            "support_surface_discarded": 0,
            "ambiguous_cluster_discarded": 0,
        }
        self.annotated_image_support_surface = None

        with timing_context("YOLO+Segmentation+FastSAM", self):
            # Run FastSAM
            if self.cfg.use_fastsam:
                fastsam_thread = threading.Thread(
                    target=self.process_fastsam, args=(color,)
                )
                fastsam_thread.start()

            # Run YOLO and SAM
            self.process_yolo_and_sam(color)

            # Waiting for FastSAM to finish
            if self.cfg.use_fastsam:
                fastsam_thread.join()

        with timing_context("Detection Filter", self):
            raw_detections = self.curr_detections
            self.last_detection_stats["yolo_base_detections"] = len(raw_detections.xyxy)

            if (
                self.cfg.use_fastsam
                and isinstance(self.fastsam_detections, sv.Detections)
                and len(self.fastsam_detections.xyxy) > 0
            ):
                raw_detections, accepted_extra = self.add_extra_detections_from_fastsam(
                    color, self.fastsam_detections, raw_detections
                )
                self.last_detection_stats["fastsam_extra_accepted"] = int(accepted_extra)

            self.filter.update_detections(raw_detections, color)
            filtered_detections = self.filter.run_filter()

        if filtered_detections is None or self.filter.get_len() == 0:
            logger.warning(
                "[Detector] No valid detections in curr frame after filtering."
            )
            self.curr_results = {}
            return

        with timing_context("CLIP+Create Object Pointcloud", self):
            cluster_thread = threading.Thread(
                target=self.process_masks, args=(filtered_detections.mask,)
            )
            cluster_thread.start()

            with timing_context("CLIP", self):
                image_crops, image_feats, text_feats = (
                    self.compute_clip_features_batched(
                        color,
                        filtered_detections,
                        self.clip_model,
                        self.clip_tokenizer,
                        self.clip_preprocess,
                        self.cfg.device,
                        self.obj_classes.get_classes_arr(),
                    )
                )

            cluster_thread.join()

        class_id = filtered_detections.class_id.copy()
        (
            class_id,
            text_feats,
            semantic_confidence,
            label_source,
            relabel_top1_scores,
            relabel_top2_scores,
            relabel_top1_class_ids,
            relabel_top2_class_ids,
        ) = self.relabel_unknown_with_clip(
            class_ids=class_id,
            confidences=filtered_detections.confidence,
            image_feats=image_feats,
            text_feats=text_feats,
        )
        filtered_detections.class_id = class_id
        self.log_relabel_summary(
            label_source=label_source,
            num_detections=len(class_id),
        )

        (
            filtered_detections,
            image_feats,
            text_feats,
            semantic_confidence,
            label_source,
            relabel_top1_scores,
            relabel_top2_scores,
            relabel_top1_class_ids,
            relabel_top2_class_ids,
            non_trackable_mask,
        ) = self.apply_support_surface_filter(
            color=color,
            detections=filtered_detections,
            image_feats=image_feats,
            text_feats=text_feats,
            semantic_confidence=semantic_confidence,
            label_source=label_source,
            relabel_top1_scores=relabel_top1_scores,
            relabel_top2_scores=relabel_top2_scores,
            relabel_top1_class_ids=relabel_top1_class_ids,
            relabel_top2_class_ids=relabel_top2_class_ids,
        )

        logger.debug(
            "[Detector][KeyframeStats] yolo_base=%d fastsam_extra_accepted=%d "
            "support_surface_discarded=%d ambiguous_cluster_discarded=%d",
            self.last_detection_stats["yolo_base_detections"],
            self.last_detection_stats["fastsam_extra_accepted"],
            self.last_detection_stats["support_surface_discarded"],
            self.last_detection_stats["ambiguous_cluster_discarded"],
        )

        results = {
            # SAM Info
            "xyxy": filtered_detections.xyxy,
            "confidence": filtered_detections.confidence,
            "class_id": class_id,
            "masks": filtered_detections.mask,
            # CLIP info
            "image_feats": image_feats,
            "text_feats": text_feats,
            "semantic_confidence": semantic_confidence,
            "label_source": label_source,
            "non_trackable": non_trackable_mask,
        }

        if self.cfg.clip_unknown_relabel.save_debug_scores:
            results["relabel_top1_scores"] = relabel_top1_scores
            results["relabel_top2_scores"] = relabel_top2_scores
            results["relabel_top1_class_ids"] = relabel_top1_class_ids
            results["relabel_top2_class_ids"] = relabel_top2_class_ids

        if self.cfg.visualize_detection:
            with timing_context("Visualize Detection", self):
                # Final post-CLIP image with all detections.
                annotated_image, _ = visualize_result_rgb(
                    color, filtered_detections, self.obj_classes.get_classes_arr()
                )
                self.annotated_image = annotated_image
                # Debug image with only FastSAM detections promoted by CLIP.
                self.annotated_image_clip_relabel = (
                    self.build_clip_relabel_debug_image(
                        color=color,
                        detections=filtered_detections,
                        label_source=label_source,
                        semantic_confidence=semantic_confidence,
                    )
                )

        self.curr_results = results

    def process_masks(self, masks):
        """
        Processes the given masks to extract and refine 3D points and colors.

        Args:
            self: The object containing configuration and data attributes.
            masks: A NumPy array of shape (N, H, W), where N is the number of masks.

        Returns:
            refined_points_list: A list of refined 3D points for each mask.
            refined_colors_list: A list of refined colors corresponding to the points for each mask.
        """

        with timing_context("Create Object Pointcloud", self):
            N, H, W = masks.shape
            
            # Initialize results to the correct length for thread safety
            self.masked_points = [None] * N
            self.masked_colors = [None] * N
            self.mask_processing_meta = [None] * N

            # Convert input data to tensors
            depth_tensor = (
                torch.from_numpy(self.curr_data.depth)
                .to(self.cfg.device)
                .float()
                .squeeze()
            )
            intrinsic_tensor = (
                torch.from_numpy(self.curr_data.intrinsics).to(self.cfg.device).float()
            )
            image_rgb_tensor = (
                torch.from_numpy(self.curr_data.color).to(self.cfg.device).float()
                / 255.0
            )
            
            # Batch masks to avoid massive intermediate (N, H, W, 3) tensors
            batch_size = self.cfg.clip.pcd_batch_size
            
            for b in range(0, N, batch_size):
                end_idx = min(b + batch_size, N)
                masks_batch = torch.from_numpy(masks[b:end_idx]).to(self.cfg.device).float()
                
                try:
                    # Generate 3D points and colors for the batch of masks
                    batch_points, batch_colors = mask_depth_to_points(
                        depth_tensor,
                        image_rgb_tensor,
                        intrinsic_tensor,
                        masks_batch,
                        self.cfg.device,
                    )
                    
                    # Process each mask in the batch
                    for i in range(batch_points.shape[0]):
                        msg_idx = b + i
                        mask_points = batch_points[i]
                        mask_colors = batch_colors[i]

                        # Filter valid points based on Z-axis > 0
                        valid_points_mask = mask_points[:, :, 2] > 0

                        if torch.sum(valid_points_mask) < self.cfg.min_points_threshold:
                            continue

                        valid_points = mask_points[valid_points_mask]
                        valid_colors = mask_colors[valid_points_mask]

                        # Early distance filter (avoid expensive refinement for distant objects)
                        if valid_points.shape[0] > 0:
                            centroid = torch.mean(valid_points, dim=0)
                            # Distance from camera origin (0,0,0) in camera frame
                            dist = torch.norm(centroid).item()
                            if dist > self.cfg.max_detection_distance:
                                continue

                        # Random sampling based on sample ratio
                        sample_ratio = self.cfg.pcd_sample_ratio
                        num_points = valid_points.shape[0]

                        if sample_ratio < 1.0:
                            sample_count = int(num_points * sample_ratio)
                            sample_indices = torch.randperm(num_points)[:sample_count]
                            downsampled_points = valid_points[sample_indices]
                            downsampled_colors = valid_colors[sample_indices]
                        else:
                            downsampled_points = valid_points
                            downsampled_colors = valid_colors

                        # Refine points using clustering
                        discard_ambiguous = bool(
                            getattr(self.cfg, "discard_ambiguous_clusters", False)
                        )
                        if discard_ambiguous:
                            (
                                refined_points,
                                refined_colors,
                                refine_meta,
                            ) = refine_points_with_clustering(
                                downsampled_points,
                                downsampled_colors,
                                eps=self.cfg.dbscan_eps,
                                min_points=self.cfg.dbscan_min_points,
                                return_metadata=True,
                            )
                            self.mask_processing_meta[msg_idx] = refine_meta
                            if refine_meta.get("ambiguous", False):
                                self.last_detection_stats["ambiguous_cluster_discarded"] += 1
                                continue
                        else:
                            refined_points, refined_colors = refine_points_with_clustering(
                                downsampled_points,
                                downsampled_colors,
                                eps=self.cfg.dbscan_eps,
                                min_points=self.cfg.dbscan_min_points,
                            )
                            self.mask_processing_meta[msg_idx] = {
                                "ambiguous": False,
                                "num_clusters": 0,
                            }

                        self.masked_points[msg_idx] = refined_points
                        self.masked_colors[msg_idx] = refined_colors
                        
                except torch.OutOfMemoryError:
                    logger.warning(f"[Detector] OOM encountered while processing masks {b} to {end_idx}. Skipping these masks.")
                    torch.cuda.empty_cache()
                finally:
                    # Clean up batch-specific tensors
                    if 'masks_batch' in locals(): del masks_batch
                    if 'batch_points' in locals(): del batch_points
                    if 'batch_colors' in locals(): del batch_colors
                    torch.cuda.empty_cache()


    def compute_max_cos_sim(self, image_feats, class_feats):
        """
        Compute the cosine similarity between image_feats and class_feats, and return the class index with the maximum similarity for each image_feat.

        Args:
            image_feats (np.ndarray): CLIP features of all current images, shape (N, 512).
            class_feats (np.ndarray): CLIP features of classes, shape (C, 512).

        Returns:
            max_indices (np.ndarray): The class index with the maximum cosine similarity for each image_feat, shape (N,).
        """
        # Normalize the features to compute cosine similarity
        image_feats_norm = image_feats / np.linalg.norm(
            image_feats, axis=1, keepdims=True
        )  # Normalize image_feats
        class_feats_norm = class_feats / np.linalg.norm(
            class_feats, axis=1, keepdims=True
        )  # Normalize class_feats

        # Compute cosine similarity: (N, 512) @ (512, C) -> (N, C)
        cos_sim = np.dot(image_feats_norm, class_feats_norm.T)

        # Find the index of the maximum similarity for each image
        max_indices = np.argmax(cos_sim, axis=1)  # shape (N,)

        return max_indices

    def depth_to_point_cloud(self, sample_rate=1) -> o3d.geometry.PointCloud:
        """
        Convert depth image to a point cloud and transform it to world coordinates.

        Parameters:
        - sample_rate: The downsampling rate for pixel selection. (1 means all pixels, 2 means every other pixel)

        Returns:
        - point_cloud: The point cloud in world coordinates as an Open3D PointCloud object.
        """
        # Extract necessary data from curr_data
        depth = self.curr_data.depth.squeeze(
            -1
        )  # Remove the last dimension if depth is (H, W, 1)
        intrinsics = self.curr_data.intrinsics
        pose = self.curr_data.pose

        # Mask out invalid depth values (e.g., depth = 0 or NaN)
        valid_mask = (depth > 0) & (
            depth != np.inf
        )  # Create a mask for valid depth values
        depth = depth[valid_mask]  # Only keep valid depth values

        # Get the corresponding u, v coordinates for valid pixels
        height, width = self.curr_data.depth.shape[:2]
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        u = u[valid_mask]
        v = v[valid_mask]

        # Downsample the points if needed (sampling every `sample_rate` pixels)
        u = u[::sample_rate]
        v = v[::sample_rate]
        depth = depth[::sample_rate]

        # Use the intrinsic matrix to convert from pixel coordinates to camera coordinates (X, Y, Z)
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Convert from pixel coordinates to normalized camera coordinates
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        # Stack the coordinates to form the point cloud in the camera coordinate system
        points_camera = np.vstack((x, y, z)).T

        # Convert points to homogeneous coordinates (4D) for transformation
        points_homogeneous = np.hstack(
            (points_camera, np.ones((points_camera.shape[0], 1)))
        )

        # Apply the pose transformation to move points to world coordinates
        points_world_homogeneous = (pose @ points_homogeneous.T).T

        # Discard the homogeneous coordinate (last column) to get the final 3D points in world coordinates
        points_world = points_world_homogeneous[:, :3]

        # Create a PointCloud object and set its points
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_world)

        return point_cloud

    def save_detection_results(
        self,
    ) -> None:

        if self.curr_results == {}:
            logger.error("[Detector] No detection, Nothing to save")
            return

        output_det_path = self.detection_path / self.curr_data.color_name
        output_det_path.mkdir(exist_ok=True, parents=True)

        # save results
        for key, value in self.curr_results.items():
            save_path = Path(output_det_path) / f"{key}"
            if isinstance(value, np.ndarray):
                # Save NumPy arrays using .npz for efficient storage
                np.savez_compressed(f"{save_path}.npz", value)
            else:
                # For other types, fall back to pickle
                with gzip.open(f"{save_path}.pkl.gz", "wb") as f:
                    pickle.dump(value, f)

        # save annotated images
        output_file_path = (
            self.detection_path / "vis" / (self.curr_data.color_name + "_annotated.jpg")
        )
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        image = cv2.cvtColor(self.curr_data.color, cv2.COLOR_BGR2RGB)

        detections = sv.Detections(
            xyxy=self.curr_results["xyxy"],
            confidence=self.curr_results["confidence"],
            class_id=self.curr_results["class_id"],
            mask=self.curr_results["masks"],
        )
        annotated_image, _ = visualize_result_rgb(
            image, detections, self.obj_classes.get_classes_arr()
        )

        self.annotated_image = annotated_image

        cv2.imwrite(str(output_file_path), annotated_image)

        if (
            self.annotated_image_support_surface is not None
            and self.last_detection_stats.get("support_surface_discarded", 0) > 0
        ):
            support_output_path = (
                self.detection_path
                / "vis"
                / (self.curr_data.color_name + "_support_surface.jpg")
            )
            cv2.imwrite(
                str(support_output_path),
                self.annotated_image_support_surface,
            )

    def load_detection_results(
        self,
    ):
        det_path = self.detection_path / self.curr_data.color_name

        det_path = Path(det_path)

        # if current frame has no detection in disk, return with empty dict
        if not det_path.exists():
            logger.error(f"[Detector] No detection results found in {det_path}")
            self.curr_results = {}
            return

        # Load results from disk
        logger.info(f"[Detector] Loading detection results from {det_path}")

        loaded_detections = {}

        for file_path in det_path.iterdir():
            # handle the files with their extensions
            if file_path.suffix == ".gz" and file_path.suffixes[-2] == ".pkl":
                key = file_path.name.replace(".pkl.gz", "")
                with gzip.open(file_path, "rb") as f:
                    loaded_detections[key] = pickle.load(f)
            elif file_path.suffix == ".npz":
                loaded_detections[file_path.stem] = np.load(file_path)["arr_0"]
            elif file_path.suffix == ".jpg":
                continue
            else:
                raise ValueError(f"{file_path} is not a .pkl.gz or .npz file!")

        self.curr_results = loaded_detections
        self.ensure_detection_metadata()

    def ensure_detection_metadata(self) -> None:
        if not self.curr_results:
            return

        class_ids = self.curr_results.get("class_id")
        confidence = self.curr_results.get("confidence")

        if class_ids is None or confidence is None:
            return

        det_count = len(class_ids)

        if "semantic_confidence" not in self.curr_results:
            logger.warning(
                "[Detector] Loaded cached detections without semantic_confidence. "
                "Treating them as pre-relabel detections."
            )
            self.curr_results["semantic_confidence"] = confidence.astype(
                np.float32, copy=True
            )

        if "label_source" not in self.curr_results:
            logger.warning(
                "[Detector] Loaded cached detections without label_source. "
                "Treating them as pre-relabel detections."
            )
            label_source = np.full(det_count, "yolo", dtype="<U32")
            if self.unknown_class_id is not None:
                label_source[class_ids == self.unknown_class_id] = "fastsam_unknown"
            self.curr_results["label_source"] = label_source

        if "non_trackable" not in self.curr_results:
            self.curr_results["non_trackable"] = np.zeros(det_count, dtype=bool)

    def get_default_label_source(self) -> np.ndarray:
        class_ids = self.curr_results.get("class_id", np.empty(0, dtype=np.int64))
        label_source = np.full(len(class_ids), "yolo", dtype="<U32")
        if self.unknown_class_id is not None:
            label_source[class_ids == self.unknown_class_id] = "fastsam_unknown"
        return label_source

    def calculate_observations(
        self,
    ) -> None:
        # if no detection, just return
        if self.curr_results == {}:
            logger.warning("[Detector] No detection, Nothing to calculate observations")
            self.curr_observations = []
            return

        # Traverse all the detections
        N, _, _ = self.curr_results["masks"].shape

        # for debugging only
        # bbox_hl_mapping = []
        for i in range(N):

            if i >= len(self.masked_points) or self.masked_points[i] is None:
                continue

            if self.curr_results.get("non_trackable", np.zeros(N, dtype=bool))[i]:
                continue

            # Create pointcloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.masked_points[i])
            pcd.colors = o3d.utility.Vector3dVector(self.masked_colors[i])
            pcd.transform(self.curr_data.pose)

            # Get bbox
            bbox = safe_create_bbox(pcd)
            # bbox = pcd.get_axis_aligned_bounding_box()

            if self.cfg.filter_ceiling:
                z = bbox.get_center()[2]

                # check z is close to ceiling_height
                if abs(z - self.cfg.ceiling_height) < self.cfg.ceiling_threshold:  # 0.1
                    continue  # If z close ceiling_height， skip this observation

            # Get Mobility
            # class_name = self.obj_classes.get_classes_arr()[self.curr_results['class_id'][i]]
            # Get distance
            distance = self.get_distance(bbox, self.curr_data.pose)
            if distance > self.cfg.max_detection_distance:
                continue

            # Init observation
            curr_obs = LocalObservation()

            # Set observation info
            curr_obs.idx = self.curr_data.idx
            curr_obs.class_id = self.curr_results["class_id"][i]
            curr_obs.mask = self.curr_results["masks"][i]

            curr_obs.xyxy = self.curr_results["xyxy"][i]
            curr_obs.semantic_confidence = self.curr_results.get(
                "semantic_confidence", self.curr_results["confidence"]
            )[i]
            curr_obs.conf = curr_obs.semantic_confidence
            curr_obs.label_source = self.curr_results.get(
                "label_source",
                self.get_default_label_source(),
            )[i]
            curr_obs.non_trackable = self.curr_results.get(
                "non_trackable",
                np.zeros(N, dtype=bool),
            )[i]

            if self.cfg.use_weighted_feature:
                curr_obs.clip_ft = self.get_weighted_feature(idx=i)
            else:
                curr_obs.clip_ft = self.curr_results["image_feats"][i]

            curr_obs.pcd = pcd
            curr_obs.bbox = bbox
            curr_obs.distance = distance

            # judge if low mobility according to the clip feature
            curr_obs.is_low_mobility = self.is_low_mobility(
                curr_obs.clip_ft
            )  # , hl_debug, hl_idx
            # for debugging only
            # bbox_hl_mapping.append([self.curr_results['xyxy'][i], hl_debug, hl_idx])

            # if curr_obs classid is desk set as low mobility
            if (
                self.obj_classes.get_classes_arr()[curr_obs.class_id]
                in self.cfg.lm_examples
            ):
                curr_obs.is_low_mobility = True

            if self.cfg.save_cropped:
                whole_image = self.curr_data.color

                # crop image by xyxy
                x1, y1, x2, y2 = map(int, curr_obs.xyxy)
                cropped_image = whole_image[y1:y2, x1:x2]
                cropped_mask = curr_obs.mask[y1:y2, x1:x2].astype(np.uint8) * 255

                masked_image = cv2.bitwise_and(
                    cropped_image, cropped_image, mask=cropped_mask
                )

                curr_obs.masked_image = masked_image
                curr_obs.cropped_image = cropped_image

            # Add observation to the list
            self.curr_observations.append(curr_obs)

        self.curr_observations = self.dedupe_current_observations(
            self.curr_observations
        )

        logger.info(
            f"[Detector] Current observations num: {len(self.curr_observations)}"
        )

    @staticmethod
    def compute_clip_cosine_similarity(clip_a: np.ndarray, clip_b: np.ndarray) -> float:
        clip_a = np.asarray(clip_a, dtype=np.float32)
        clip_b = np.asarray(clip_b, dtype=np.float32)
        if clip_a.size == 0 or clip_b.size == 0:
            return 0.0
        denom = np.linalg.norm(clip_a) * np.linalg.norm(clip_b)
        if denom <= 1e-8:
            return 0.0
        return float(np.dot(clip_a, clip_b) / denom)

    @staticmethod
    def compute_bbox_xy_overlap_ratio(bbox_a, bbox_b) -> float:
        min_a = np.asarray(bbox_a.get_min_bound(), dtype=np.float32)
        max_a = np.asarray(bbox_a.get_max_bound(), dtype=np.float32)
        min_b = np.asarray(bbox_b.get_min_bound(), dtype=np.float32)
        max_b = np.asarray(bbox_b.get_max_bound(), dtype=np.float32)

        inter_min = np.maximum(min_a[:2], min_b[:2])
        inter_max = np.minimum(max_a[:2], max_b[:2])
        inter_dims = np.maximum(inter_max - inter_min, 0.0)
        inter_area = inter_dims[0] * inter_dims[1]

        area_a = max((max_a[0] - min_a[0]) * (max_a[1] - min_a[1]), 1e-8)
        area_b = max((max_b[0] - min_b[0]) * (max_b[1] - min_b[1]), 1e-8)
        return float(max(inter_area / area_a, inter_area / area_b))

    @staticmethod
    def compute_bbox_3d_intersection(bbox_a, bbox_b) -> float:
        min_a = np.asarray(bbox_a.get_min_bound(), dtype=np.float32)
        max_a = np.asarray(bbox_a.get_max_bound(), dtype=np.float32)
        min_b = np.asarray(bbox_b.get_min_bound(), dtype=np.float32)
        max_b = np.asarray(bbox_b.get_max_bound(), dtype=np.float32)

        inter_min = np.maximum(min_a, min_b)
        inter_max = np.minimum(max_a, max_b)
        inter_dims = np.maximum(inter_max - inter_min, 0.0)
        return float(np.prod(inter_dims))

    def should_dedupe_observation_pair(self, obs_a, obs_b) -> bool:
        class_a_unknown = obs_a.class_id == self.unknown_class_id
        class_b_unknown = obs_b.class_id == self.unknown_class_id

        if not class_a_unknown and not class_b_unknown and obs_a.class_id != obs_b.class_id:
            return False

        clip_cos = self.compute_clip_cosine_similarity(obs_a.clip_ft, obs_b.clip_ft)
        if clip_cos < 0.35:
            return False

        center_a = np.asarray(obs_a.bbox.get_center(), dtype=np.float32)
        center_b = np.asarray(obs_b.bbox.get_center(), dtype=np.float32)
        if np.linalg.norm(center_a[:2] - center_b[:2]) > 0.03:
            return False

        bbox_overlap = self.compute_bbox_xy_overlap_ratio(obs_a.bbox, obs_b.bbox)
        if bbox_overlap >= 0.25:
            return True

        return self.compute_bbox_3d_intersection(obs_a.bbox, obs_b.bbox) > 0.0

    def get_observation_keep_priority(self, obs) -> tuple:
        is_known = int(obs.class_id != self.unknown_class_id)
        is_yolo = int(getattr(obs, "label_source", "") == "yolo")
        semantic_conf = float(getattr(obs, "semantic_confidence", 0.0))
        point_count = len(getattr(obs.pcd, "points", []))
        return (is_known, is_yolo, semantic_conf, point_count)

    def dedupe_current_observations(self, observations: list) -> list:
        if len(observations) <= 1:
            return observations

        sorted_indices = sorted(
            range(len(observations)),
            key=lambda idx: self.get_observation_keep_priority(observations[idx]),
            reverse=True,
        )
        keep_indices = []
        discarded = 0

        for obs_idx in sorted_indices:
            obs = observations[obs_idx]
            is_duplicate = False
            for kept_idx in keep_indices:
                if self.should_dedupe_observation_pair(obs, observations[kept_idx]):
                    is_duplicate = True
                    discarded += 1
                    break
            if not is_duplicate:
                keep_indices.append(obs_idx)

        keep_indices = set(keep_indices)
        deduped = [
            obs for idx, obs in enumerate(observations) if idx in keep_indices
        ]

        if discarded > 0:
            logger.info(
                "[Detector] Intra-frame dedupe removed %d duplicated observations.",
                discarded,
            )

        return deduped

    def get_weighted_feature(self, idx):
        image_feat = self.curr_results["image_feats"][idx]
        text_feat = self.curr_results["text_feats"][idx]

        w_image = self.cfg.image_weight
        w_text = 1 - w_image

        weighted_feature = w_image * image_feat + w_text * text_feat

        norm = np.linalg.norm(weighted_feature)
        if norm > 0:
            weighted_feature /= norm

        return weighted_feature

    def visualize_time(
        self,
        elapsed_time,
    ) -> None:
        logger.info(f"[Detector][Visualize] Elapsed time: {elapsed_time:.4f} seconds")
        self.visualizer.log(
            "plot_time/frame_elapsed_time",
            self.visualizer.Scalar(elapsed_time),
            self.visualizer.SeriesLine(width=2.5, color=[255, 0, 0]),  # Red color
        )

    def visualize_memory(
        self,
        memory_usage,
    ) -> None:
        logger.info(f"[Detector][Visualize] Memory usage: {memory_usage:.2f} MB")
        self.visualizer.log(
            "plot_memory/memory_usage",
            self.visualizer.Scalar(memory_usage),
            self.visualizer.SeriesLine(width=2.5, color=[0, 255, 0]),  # Green color
        )

    def visualize_detection(
        self,
    ) -> None:

        if self.annotated_image is not None:
            self.visualizer.log(
                "world/camera/rgb_image_annotated",
                self.visualizer.Image(self.annotated_image),
            )

        if self.cfg.show_local_entities:
            self.visualizer.log(
                "world/camera_raw/rgb_image",
                self.visualizer.Image(self.curr_data.color),
            )

            if self.annotated_image_fs is not None:
                self.visualizer.log(
                    "world/camera_fs/rgb_image_annotated",
                    self.visualizer.Image(self.annotated_image_fs),
                )

            if self.annotated_image_fs_after is not None:
                self.visualizer.log(
                    "world/camera_fs_after/rgb_image_annotated",
                    self.visualizer.Image(self.annotated_image_fs_after),
                )

        # Visualize camera traj
        focal_length = [
            self.curr_data.intrinsics[0, 0].item(),
            self.curr_data.intrinsics[1, 1].item(),
        ]
        principal_point = [
            self.curr_data.intrinsics[0, 2].item(),
            self.curr_data.intrinsics[1, 2].item(),
        ]
        height, width = self.curr_data.color.shape[:2]
        resolution = [width, height]
        self.visualizer.log(
            "world/camera",
            self.visualizer.Pinhole(
                resolution=resolution,
                focal_length=focal_length,
                principal_point=principal_point,
            ),
        )

        translation = self.curr_data.pose[:3, 3].tolist()

        # change the rotation mat to axis-angle
        axis, angle = self.visualizer.rotation_matrix_to_axis_angle(
            self.curr_data.pose[:3, :3]
        )
        self.visualizer.log(
            "world/camera",
            self.visualizer.Transform3D(
                translation=translation,
                rotation=self.visualizer.RotationAxisAngle(axis=axis, angle=angle),
                from_parent=False,
            ),
        )

        # follower camera for recording
        # Visualize camera traj
        f = [
            self.curr_data.intrinsics[0, 0].item(),
            self.curr_data.intrinsics[1, 1].item(),
        ]
        p = [
            self.curr_data.intrinsics[0, 2].item(),
            self.curr_data.intrinsics[1, 2].item(),
        ]
        h, w = self.curr_data.color.shape[:2]
        r = [w, h]
        self.visualizer.log(
            "world/follower_camera",
            self.visualizer.Pinhole(resolution=r, focal_length=f, principal_point=p),
        )
        self.visualizer.log(
            "world/follower_camera_2",
            self.visualizer.Pinhole(resolution=r, focal_length=f, principal_point=p),
        )

        pose_current = self.curr_data.pose
        cam2_to_cam1 = self.create_camera2_to_camera1_transform()
        cam2_to_cam1_2 = self.create_camera2_to_camera1_transform2()
        pose_new = pose_current @ cam2_to_cam1
        pose_new_2 = pose_current @ cam2_to_cam1_2

        translation = pose_new[:3, 3].tolist()
        # change the rotation mat to axis-angle
        axis, angle = self.visualizer.rotation_matrix_to_axis_angle(pose_new[:3, :3])
        self.visualizer.log(
            "world/follower_camera",
            self.visualizer.Transform3D(
                translation=translation,
                rotation=self.visualizer.RotationAxisAngle(axis=axis, angle=angle),
                from_parent=False,
            ),
        )

        # Using for visualization
        pose_smooth = self.pose_filter_follower.update(pose_new_2)
        translation = pose_smooth[:3, 3].tolist()
        axis, angle = self.visualizer.rotation_matrix_to_axis_angle(pose_smooth[:3, :3])

        self.visualizer.log(
            "world/follower_camera_2",
            self.visualizer.Transform3D(
                translation=translation,
                rotation=self.visualizer.RotationAxisAngle(axis=axis, angle=angle),
                from_parent=False,
            ),
        )

        if self.prev_data is not None:
            prev_translation = self.prev_data.pose[:3, 3].tolist()
            prev_quaternion = self.visualizer.rotation_matrix_to_quaternion(
                self.prev_data.pose[:3, :3]
            )

            # # Log a line strip from the previous to the current camera pose
            # self.visualizer.log(
            #     f"world/camera_trajectory/{self.curr_data.idx}",
            #     self.visualizer.LineStrips3D(
            #         [np.vstack([prev_translation, translation]).tolist()],
            #         colors=[[255, 0, 0]]  # Red color for the trajectory line
            #     )
            # )

        if self.cfg.show_debug_entities:
            layout_pointcloud = self.get_layout_pointcloud()
            positions = layout_pointcloud.points
            pcd_entity = "world/layout"
            self.visualizer.log(pcd_entity, self.visualizer.Points3D(positions))

    def create_camera2_to_camera1_transform(self):
        # Define translation vector: translation of camera 2 relative to camera 1
        translation = np.array(self.cfg.follower_translation)  # Up 0.2m, back -0.2m

        # Rotation angles in degrees
        angle_roll = self.cfg.follower_roll  # Rotation around X axis
        angle_pitch = self.cfg.follower_pitch  # Rotation around Y axis
        angle_yaw = self.cfg.follower_yaw  # Rotation around Z axis

        # Convert angles to radians
        angle_roll_rad = np.radians(angle_roll)
        angle_pitch_rad = np.radians(angle_pitch)
        angle_yaw_rad = np.radians(angle_yaw)

        # Rotation matrix around X axis (roll)
        rotation_roll = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle_roll_rad), -np.sin(angle_roll_rad)],
                [0, np.sin(angle_roll_rad), np.cos(angle_roll_rad)],
            ]
        )

        # Rotation matrix around Y axis (pitch)
        rotation_pitch = np.array(
            [
                [np.cos(angle_pitch_rad), 0, np.sin(angle_pitch_rad)],
                [0, 1, 0],
                [-np.sin(angle_pitch_rad), 0, np.cos(angle_pitch_rad)],
            ]
        )

        # Rotation matrix around Z axis (yaw)
        rotation_yaw = np.array(
            [
                [np.cos(angle_yaw_rad), -np.sin(angle_yaw_rad), 0],
                [np.sin(angle_yaw_rad), np.cos(angle_yaw_rad), 0],
                [0, 0, 1],
            ]
        )

        # Combined rotation matrix: order is Z, Y, X
        rotation_matrix = rotation_yaw @ rotation_pitch @ rotation_roll

        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix  # Fill rotation matrix
        transform[:3, 3] = translation  # Fill translation vector

        return transform

    def create_camera2_to_camera1_transform2(self):
        # Define translation vector: translation of camera 2 relative to camera 1
        translation = np.array(self.cfg.follower_translation2)  # Up 0.2m, back -0.2m

        # Rotation angles in degrees
        angle_roll = self.cfg.follower_roll2  # Rotation around X axis
        angle_pitch = self.cfg.follower_pitch2  # Rotation around Y axis
        angle_yaw = self.cfg.follower_yaw2  # Rotation around Z axis

        # Convert angles to radians
        angle_roll_rad = np.radians(angle_roll)
        angle_pitch_rad = np.radians(angle_pitch)
        angle_yaw_rad = np.radians(angle_yaw)

        # Rotation matrix around X axis (roll)
        rotation_roll = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle_roll_rad), -np.sin(angle_roll_rad)],
                [0, np.sin(angle_roll_rad), np.cos(angle_roll_rad)],
            ]
        )

        # Rotation matrix around Y axis (pitch)
        rotation_pitch = np.array(
            [
                [np.cos(angle_pitch_rad), 0, np.sin(angle_pitch_rad)],
                [0, 1, 0],
                [-np.sin(angle_pitch_rad), 0, np.cos(angle_pitch_rad)],
            ]
        )

        # Rotation matrix around Z axis (yaw)
        rotation_yaw = np.array(
            [
                [np.cos(angle_yaw_rad), -np.sin(angle_yaw_rad), 0],
                [np.sin(angle_yaw_rad), np.cos(angle_yaw_rad), 0],
                [0, 0, 1],
            ]
        )

        # Combined rotation matrix: order is Z, Y, X
        rotation_matrix = rotation_yaw @ rotation_pitch @ rotation_roll

        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix  # Fill rotation matrix
        transform[:3, 3] = translation  # Fill translation vector

        return transform

    def is_low_mobility(self, clip_feat) -> bool:
        # Calculate the cosine similarity between the clip feature and the prototypes
        clip_feat = clip_feat.reshape(1, -1)
        sim = cosine_similarity(clip_feat, self.proto_feats)
        sim = sim.reshape(-1)

        sim_lm = np.max(sim[: self.num_examples[0]])
        sim_hm = np.max(
            sim[self.num_examples[0] : (self.num_examples[0] + self.num_examples[1])]
        )
        sim_lm_des = np.max(sim[(self.num_examples[0] + self.num_examples[1]) :])

        # for debugging only
        # lm_idx = np.argmax(sim[:self.num_examples[0]])
        # hm_idx = np.argmax(sim[self.num_examples[0] : (self.num_examples[0]+self.num_examples[1])])
        # lm_des_idx = np.argmax(sim[(self.num_examples[0]+self.num_examples[1]):])
        # hl_debug = np.array([sim_lm, sim_hm, sim_lm_des])
        # hl_idx = np.array([lm_idx, hm_idx, lm_des_idx])

        # Use configurable thresholds
        similarity_delta = self.cfg.mobility.similarity_delta
        descriptor_threshold = self.cfg.mobility.descriptor_threshold
        
        if sim_lm > sim_hm + similarity_delta:
            res = True
        elif sim_lm + similarity_delta < sim_hm:
            res = False
        else:
            res = sim_lm_des > descriptor_threshold
        return res  # , hl_debug, hl_idx

    def get_distance(self, bbox, pose) -> float:
        # Get the center of the bounding box
        bbox_center = np.array(bbox.get_center())

        # Get the translation part of the pose (assuming it's a 4x4 matrix)
        pose_translation = np.array(pose[:3, 3])  # Extract translation (x, y, z)

        # Calculate the Euclidean distance between the pose translation and the bbox center
        distance = np.linalg.norm(bbox_center - pose_translation)

        return distance

    def compute_clip_features_batched(
        self,
        image,
        detections,
        clip_model,
        clip_tokenizer,
        clip_preprocess,
        device,
        classes,
    ):
        # Convert the image to a PIL Image
        image = Image.fromarray(image)

        # Set the padding for cropping
        padding = 20

        # Initialize lists to store the cropped images and features
        image_crops = []
        image_feats = []
        text_feats = []

        # Initialize lists to store preprocessed images and text tokens for batch processing
        preprocessed_images = []
        text_tokens = []

        # Prepare data for batch processing
        for idx in range(len(detections.xyxy)):
            x_min, y_min, x_max, y_max = detections.xyxy[idx]
            image_width, image_height = image.size

            # Calculate the padding for each side of the bounding box
            left_padding = min(padding, x_min)
            top_padding = min(padding, y_min)
            right_padding = min(padding, image_width - x_max)
            bottom_padding = min(padding, image_height - y_max)

            # Adjust the bounding box coordinates based on the padding
            x_min -= left_padding
            y_min -= top_padding
            x_max += right_padding
            y_max += bottom_padding

            # Crop the image
            cropped_image = image.crop((x_min, y_min, x_max, y_max))

            # Preprocess the cropped image
            preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0)
            preprocessed_images.append(preprocessed_image)

            # Get the class id for the detection
            class_id = detections.class_id[idx]

            # Append the class name to the text tokens list
            text_tokens.append(classes[class_id])

            # Append the cropped image to the image crops list
            image_crops.append(cropped_image)

        # Convert lists to batches
        batch_size = self.cfg.clip.clip_batch_size
        num_detections = len(preprocessed_images)
        
        all_image_features = []
        all_text_features = []

        # Perform batch inference
        with torch.no_grad():
            for i in range(0, num_detections, batch_size):
                end_idx = min(i + batch_size, num_detections)
                
                # Batch processing for images
                images_chunk = torch.cat(preprocessed_images[i:end_idx], dim=0).to(device)
                image_features_chunk = clip_model.encode_image(images_chunk)
                image_features_chunk /= image_features_chunk.norm(dim=-1, keepdim=True)
                all_image_features.append(image_features_chunk.cpu())
                
                # Batch processing for text
                text_chunk = clip_tokenizer(text_tokens[i:end_idx]).to(device)
                text_features_chunk = clip_model.encode_text(text_chunk)
                text_features_chunk /= text_features_chunk.norm(dim=-1, keepdim=True)
                all_text_features.append(text_features_chunk.cpu())
                
                # Explicitly clear VRAM after each batch
                del images_chunk, image_features_chunk, text_chunk, text_features_chunk
                torch.cuda.empty_cache()

        # Concatenate all features
        image_features = torch.cat(all_image_features, dim=0)
        text_features = torch.cat(all_text_features, dim=0)

        # Convert the image and text features to numpy arrays
        image_feats = image_features.numpy()
        text_feats = text_features.numpy()

        if self.cfg.use_avg_feat_for_unknown and self.unknown_class_id is not None:
            count = 0
            for idx, class_id in enumerate(detections.class_id):
                if class_id == self.unknown_class_id:
                    count += 1
                    # Modify the text_feats for the unknown class
                    text_feats[idx] = (
                        self.class_feats_mean
                    )  # You can modify how you update the text_feats here

                    # random_feats = np.random.rand(*self.class_feats_mean.shape)
                    # random_feats /=     np.linalg.norm(random_feats)

                    # text_feats[idx] = random_feats

            logger.info(
                f"[Detector] Updated {count} unknown class text features to the mean value."
            )
        else:
            count = 0
            for idx, class_id in enumerate(detections.class_id):
                if self.unknown_class_id is not None and class_id == self.unknown_class_id:
                    count += 1
                    # Modify the text_feats for the unknown class
                    # text_feats[idx] = self.class_feats_mean  # You can modify how you update the text_feats here

                    random_feats = np.random.rand(*self.class_feats_mean.shape)
                    random_feats /= np.linalg.norm(random_feats)

                    text_feats[idx] = random_feats

        # Return the cropped images, image features, and text features
        return image_crops, image_feats, text_feats


class Filter:
    def __init__(
        self,
        classes,
        iou_th: float = 0.80,
        proximity_th: float = 0.95,
        keep_larger: bool = True,
        small_mask_size: int = 200,
        skip_refinement: bool = False,
        coarse_mask_ratio: float = 2.5,
        unknown_overlap_containment: float = 0.9,
    ):

        self.confidence = None
        self.class_id = None
        self.xyxy = None
        self.masks = None
        self.color = None
        self.masks_size = None
        self.inter_np = None

        self.skip_refinement = skip_refinement
        self.classes = classes
        self.iou_th = iou_th
        self.proximity_th = proximity_th
        self.keep_larger = keep_larger
        self.small_mask_size = small_mask_size
        self.coarse_mask_ratio = coarse_mask_ratio
        self.unknown_overlap_containment = unknown_overlap_containment

        self.device = "cpu"

    def update_detections(self, detections: sv.Detections, color: np.array):
        with timing_context("update_detections", self):
            self.color = color

            self.confidence = detections.confidence
            self.class_id = detections.class_id
            self.xyxy = detections.xyxy
            self.masks = detections.mask
            self.recompute_filter_metadata()

    def set_device(self, device):
        self.device = device

    def recompute_filter_metadata(self):
        """Keep mask sizes, bboxes, and pairwise intersections in sync."""
        if self.confidence is None:
            self.masks_size = np.zeros((0,), dtype=np.int64)
            self.inter_np = np.zeros((0, 0), dtype=np.float32)
            return

        self.masks = np.asarray(self.masks, dtype=np.bool_)
        N = self.get_len()
        if N == 0:
            self.masks_size = np.zeros((0,), dtype=np.int64)
            self.inter_np = np.zeros((0, 0), dtype=np.float32)
            return

        if self.xyxy is None or len(self.xyxy) != N:
            self.xyxy = np.zeros((N, 4), dtype=np.float32)
        else:
            self.xyxy = np.asarray(self.xyxy, dtype=np.float32)

        self.masks_size = np.sum(self.masks, axis=(1, 2)).astype(np.int64)
        for idx in range(N):
            bbox = update_bbox(self.masks[idx])
            if bbox is None:
                self.xyxy[idx] = np.zeros((4,), dtype=np.float32)
            else:
                self.xyxy[idx] = np.asarray(bbox, dtype=np.float32)

        device = self.device
        masks = torch.tensor(self.masks, dtype=torch.float32).to(device)
        intersection = torch.matmul(masks.view(N, -1), masks.view(N, -1).T)
        self.inter_np = intersection.cpu().numpy()

    def run_filter(self):
        original_num = self.get_len()
        if self.confidence is None or original_num == 0:
            logger.warning("[Detector][Filter] No detections to filter.")
            return None

        keep = self.filter_by_mask_size()
        self.set_detections(keep)

        if not self.skip_refinement:
            keep = self.filter_by_iou()
            self.set_detections(keep)

            keep = self.filter_by_proximity()
            self.set_detections(keep)

            self.overlap_check()

            keep = self.filter_by_mask_size()
            self.set_detections(keep)

        keep = self.filter_by_bg()
        self.set_detections(keep)

        keep = self.filter_by_mask_size()
        self.set_detections(keep)

        if self.get_len() == 0:
            logger.warning(
                "[Detector][Filter] After filtering, no detection result remains..."
            )
            return None
        logger.info(
            f"[Detector][Filter] Filtered {self.get_len()} out of {original_num}"
        )

        # create new detections object and return
        filtered_detections = sv.Detections(
            class_id=np.array(self.class_id, dtype=np.int64),
            confidence=np.array(self.confidence, dtype=np.float32),
            xyxy=np.array(self.xyxy, dtype=np.float32),
            mask=np.array(self.masks, dtype=np.bool_),
        )
        return filtered_detections

    def get_len(self):
        return len(self.confidence)

    def set_detections(self, keep):
        if len(keep) != self.get_len():
            logger.warning(
                "[Detector][Filter] The boolean list should be as long as the detections."
            )
            return

        self.confidence, self.class_id, self.xyxy, self.masks, self.masks_size = (
            self.confidence[keep],
            self.class_id[keep],
            self.xyxy[keep],
            self.masks[keep],
            self.masks_size[keep],
        )
        self.recompute_filter_metadata()

    def filter_by_iou(self):
        N = self.get_len()
        # use a list of boolean to control
        if N == 0:
            return np.array([], dtype=bool)

        masks = self.masks
        masks_size = self.masks_size

        # Compute pairwise IoU matrix using matrix operations for acceleration
        intersection = self.inter_np
        area = masks.reshape(N, -1).sum(axis=1)
        union = area[:, None] + area[None, :] - intersection
        iou_matrix = intersection / np.clip(union, 1e-7, None)

        # Initialize keep mask
        keep = np.ones(N, dtype=bool)
        unknown_class_id = self.classes.get_unknown_class_id()

        # Apply IoU threshold and keep larger/smaller masks
        for i in range(N):
            if not keep[i]:
                continue
            for j in range(i + 1, N):
                if not keep[j]:
                    continue
                i_is_unknown = (
                    unknown_class_id is not None
                    and self.class_id[i] == unknown_class_id
                )
                j_is_unknown = (
                    unknown_class_id is not None
                    and self.class_id[j] == unknown_class_id
                )
                smaller_area = max(min(area[i], area[j]), 1.0)
                containment = intersection[i, j] / smaller_area
                if i_is_unknown and j_is_unknown and (
                    containment >= self.unknown_overlap_containment
                    or iou_matrix[i, j] > self.iou_th
                ):
                    if masks_size[i] > masks_size[j] or (
                        masks_size[i] == masks_size[j]
                        and self.confidence[i] >= self.confidence[j]
                    ):
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
                    continue
                if iou_matrix[i, j] > self.iou_th:
                    size_ratio = max(
                        masks_size[i] / max(masks_size[j], 1),
                        masks_size[j] / max(masks_size[i], 1),
                    )
                    if (
                        self.class_id[i] != self.class_id[j]
                        and not (i_is_unknown or j_is_unknown)
                    ):
                        continue
                    if size_ratio > self.coarse_mask_ratio and (i_is_unknown ^ j_is_unknown):
                        continue

                    if ((masks_size[i] > masks_size[j]) and self.keep_larger) or (
                        (masks_size[i] < masks_size[j]) and not self.keep_larger
                    ):
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

        logger.info(
            f"[Detector][Filter] Original number of detections: {N}, after mask IoU filter: {np.sum(keep)}"
        )
        return keep

    def filter_by_proximity(self):
        if self.color is None:
            logger.warning("[Detector][Filter] No color image is provided.")
            return np.ones(self.get_len(), dtype=bool)
        N = self.get_len()
        if N == 0:
            return np.array([], dtype=bool)
        keep = np.ones(N, dtype=bool)
        unknown_class_id = self.classes.get_unknown_class_id()

        while True:
            self.recompute_filter_metadata()
            overlap = self.inter_np > 0
            np.fill_diagonal(overlap, False)
            masks_size = self.masks_size

            cropped_images = []
            cropped_masks = []
            for i in range(N):
                x1, y1, x2, y2 = map(int, self.xyxy[i])
                cropped_image = self.color[y1:y2, x1:x2]
                cropped_images.append(cropped_image)
                cropped_mask = self.masks[i][y1:y2, x1:x2].astype(bool)
                cropped_masks.append(cropped_mask)

            changed = False
            for i in range(N):
                if not keep[i]:
                    continue
                for j in range(i + 1, N):
                    if not keep[j] or not overlap[i, j]:
                        continue

                    from_same_dis = if_same_distribution(
                        cropped_images[i],
                        cropped_images[j],
                        cropped_masks[i],
                        cropped_masks[j],
                        self.proximity_th,
                    )

                    if not from_same_dis:
                        continue

                    size_ratio = max(
                        masks_size[i] / max(masks_size[j], 1),
                        masks_size[j] / max(masks_size[i], 1),
                    )
                    i_is_unknown = (
                        unknown_class_id is not None
                        and self.class_id[i] == unknown_class_id
                    )
                    j_is_unknown = (
                        unknown_class_id is not None
                        and self.class_id[j] == unknown_class_id
                    )
                    if (
                        self.class_id[i] != self.class_id[j]
                        and not (i_is_unknown or j_is_unknown)
                    ):
                        continue
                    if size_ratio > self.coarse_mask_ratio and (i_is_unknown ^ j_is_unknown):
                        continue

                    class_i = self.classes.get_classes_arr()[self.class_id[i]]
                    class_j = self.classes.get_classes_arr()[self.class_id[j]]
                    if ((masks_size[i] > masks_size[j]) and self.keep_larger) or (
                        (masks_size[i] < masks_size[j]) and not self.keep_larger
                    ):
                        keep[j] = False
                        self.merge_detections(j, i, recompute=True)
                        logger.info(
                            f"[Detector][Filter] Merging {class_j} into {class_i}"
                        )
                    else:
                        keep[i] = False
                        self.merge_detections(i, j, recompute=True)
                        logger.info(
                            f"[Detector][Filter] Merging {class_i} into {class_j}"
                        )
                    changed = True
                    break
                if changed:
                    break

            if not changed:
                break

        logger.info(
            f"[Detector][Filter] Original number of detections: {N}, after proximity filter: {np.sum(keep)}"
        )
        return keep

    def overlap_check(self):
        unknown_class_id = self.classes.get_unknown_class_id()

        while True:
            self.recompute_filter_metadata()
            N = self.get_len()
            if N == 0:
                return

            overlap = self.inter_np > 0
            np.fill_diagonal(overlap, False)
            overlap_pairs = np.argwhere(np.triu(overlap, k=1))
            if len(overlap_pairs) == 0:
                return

            changed = False
            for i, j in overlap_pairs:
                area_i = int(self.masks_size[i])
                area_j = int(self.masks_size[j])
                inter = float(self.inter_np[i, j])
                smaller_area = max(min(area_i, area_j), 1)
                union = max(area_i + area_j - inter, 1.0)
                containment = inter / smaller_area
                iou = inter / union

                i_is_unknown = (
                    unknown_class_id is not None
                    and self.class_id[i] == unknown_class_id
                )
                j_is_unknown = (
                    unknown_class_id is not None
                    and self.class_id[j] == unknown_class_id
                )
                if i_is_unknown and j_is_unknown and (
                    containment >= self.unknown_overlap_containment or iou >= self.iou_th
                ):
                    if area_i > area_j or (
                        area_i == area_j and self.confidence[i] >= self.confidence[j]
                    ):
                        drop_idx = j
                    else:
                        drop_idx = i
                    keep = np.ones(N, dtype=bool)
                    keep[drop_idx] = False
                    self.set_detections(keep)
                    changed = True
                    break

                if area_i >= area_j:
                    trim_idx, reference_idx = i, j
                else:
                    trim_idx, reference_idx = j, i

                trimmed_mask = self.masks[trim_idx] & (~self.masks[reference_idx])
                trimmed_bbox = update_bbox(trimmed_mask)
                if (
                    trimmed_bbox is None
                    or int(np.sum(trimmed_mask)) < self.small_mask_size
                ):
                    keep = np.ones(N, dtype=bool)
                    keep[trim_idx] = False
                    self.set_detections(keep)
                else:
                    self.masks[trim_idx] = trimmed_mask
                    self.xyxy[trim_idx] = np.asarray(trimmed_bbox, dtype=np.float32)
                    self.recompute_filter_metadata()
                changed = True
                break

            if not changed:
                return

    def filter_by_bg(self):
        N = self.get_len()
        keep = np.ones(N, dtype=bool)

        for idx, class_id in enumerate(self.class_id):
            if self.classes.get_classes_arr()[class_id] in self.classes.bg_classes:
                logger.info(
                    f"[Detector][Filter] Removing {self.classes.get_classes_arr()[class_id]} because it is a background class."
                )
                keep[idx] = False
        return keep

    def filter_by_mask_size(self):
        if self.get_len() == 0:
            return np.array([], dtype=bool)
        keep = self.masks_size >= self.small_mask_size
        for idx, is_keep in enumerate(keep):
            if not is_keep:
                class_name = self.classes.get_classes_arr()[self.class_id[idx]]
                logger.info(
                    f"[Detector][Filter] Removing {class_name} because the mask size is too small."
                )
        return keep

    def merge_detections(self, det, target, recompute=False):
        # merge det into the target detection
        self.masks[target] = np.logical_or(self.masks[target], self.masks[det])
        bbox = update_bbox(self.masks[target])
        if bbox is None:
            self.xyxy[target, :] = np.zeros((4,), dtype=np.float32)
        else:
            self.xyxy[target, :] = np.asarray(bbox, dtype=np.float32)
        if recompute:
            self.recompute_filter_metadata()


def update_bbox(mask):
    y, x = np.nonzero(mask)
    if len(x) == 0 or len(y) == 0:
        return None
    return np.min(x), np.min(y), np.max(x), np.max(y)


def if_same_distribution(img1, img2, mask1, mask2, sim_threshold):
    if (
        img1.size == 0
        or img2.size == 0
        or mask1.size == 0
        or mask2.size == 0
        or not np.any(mask1)
        or not np.any(mask2)
    ):
        return False

    # Separate the image into three channels
    b1, g1, r1 = cv2.split(img1)
    b2, g2, r2 = cv2.split(img2)

    b1, g1, r1 = b1[mask1], g1[mask1], r1[mask1]
    b2, g2, r2 = b2[mask2], g2[mask2], r2[mask2]

    # Compute histograms for each channel
    num_batches = 16
    hist_b1, _ = np.histogram(b1, bins=num_batches, range=(0, 256))
    hist_g1, _ = np.histogram(g1, bins=num_batches, range=(0, 256))
    hist_r1, _ = np.histogram(r1, bins=num_batches, range=(0, 256))
    hist_b2, _ = np.histogram(b2, bins=num_batches, range=(0, 256))
    hist_g2, _ = np.histogram(g2, bins=num_batches, range=(0, 256))
    hist_r2, _ = np.histogram(r2, bins=num_batches, range=(0, 256))

    # Normalize histograms
    hist_b1 = hist_b1 / max(np.linalg.norm(hist_b1), 1e-7)
    hist_g1 = hist_g1 / max(np.linalg.norm(hist_g1), 1e-7)
    hist_r1 = hist_r1 / max(np.linalg.norm(hist_r1), 1e-7)
    hist_b2 = hist_b2 / max(np.linalg.norm(hist_b2), 1e-7)
    hist_g2 = hist_g2 / max(np.linalg.norm(hist_g2), 1e-7)
    hist_r2 = hist_r2 / max(np.linalg.norm(hist_r2), 1e-7)

    # Concatenate histograms
    hist1 = np.concatenate([hist_b1, hist_g1, hist_r1])
    hist2 = np.concatenate([hist_b2, hist_g2, hist_r2])

    # Compute cosine similarity
    cos_sim = cosine_similarity([hist1], [hist2])[0][0]

    return cos_sim > sim_threshold


def get_text_features(
    class_names: list, clip_model, clip_tokenizer, device, clip_length, batch_size=64
) -> np.ndarray:

    multiple_templates = [
        "{}",
        "There is the {} in the scene.",
    ]

    # Get all the prompted sequences
    class_name_prompts = [
        x.format(lm) for lm in class_names for x in multiple_templates
    ]

    # Get tokens
    text_tokens = clip_tokenizer(class_name_prompts).to(device)
    # Get Output features
    text_feats = np.zeros((len(class_name_prompts), clip_length), dtype=np.float32)
    # Get the text feature batch by batch
    text_id = 0
    while text_id < len(class_name_prompts):
        # Get batch size
        batch_size = min(len(class_name_prompts) - text_id, batch_size)
        # Get text prompts based on batch size
        text_batch = text_tokens[text_id : text_id + batch_size]
        with torch.no_grad():
            batch_feats = clip_model.encode_text(text_batch).float()

        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        batch_feats = np.float32(batch_feats.cpu())
        # move the calculated batch into the Ouput features
        text_feats[text_id : text_id + batch_size, :] = batch_feats
        # Move on and Move on
        text_id += batch_size

    # shrink the output text features into classes names size
    text_feats = text_feats.reshape((-1, len(multiple_templates), text_feats.shape[-1]))
    text_feats = np.mean(text_feats, axis=1)

    # TODO: Should we do normalization? Answer should be YES
    norms = np.linalg.norm(text_feats, axis=1, keepdims=True)
    text_feats /= norms

    return text_feats


def save_hilow_debug(bbox_hl_mapping, output_image, frame_idx):
    for item in bbox_hl_mapping:
        bbox = item[0]
        hl_debug = item[1]
        hl_idx = item[2]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{hl_debug[0]:.3f}_{hl_idx[0]}, {hl_debug[1]:.3f}_{hl_idx[1]}, {hl_debug[2]:.3f}_{hl_idx[2]}"
        cv2.putText(
            output_image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    output_image_path = f"./debug/{frame_idx}_bbox_hl_mapping.jpg"
    cv2.imwrite(output_image_path, output_image)
