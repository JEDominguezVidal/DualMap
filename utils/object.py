# Standard library imports
import copy
import logging
import os
import pdb
import pickle
import uuid
from collections import Counter, deque
from enum import Enum
from typing import List, Optional

# Third-party imports
import numpy as np
import open3d as o3d
from omegaconf import DictConfig

# Local module imports
from utils.types import Observation

# Set up the module-level logger
logger = logging.getLogger(__name__)


class LocalObjStatus(Enum):
    UPDATING = "updating"
    PENDING = "pending for updating"
    ELIMINATION = "elimination"
    LM_ELIMINATION = "elimination for low mobility"
    HM_ELIMINATION = "elimination for high mpbility"
    WAITING = "waiting for stable obj process"


class BaseObject:

    # Global variable for config
    _cfg = None

    def __init__(self):
        # id
        self.uid = uuid.uuid4()

        # obs info
        self.observed_num = 0
        self.observations: List[str] = []

        # Spatial primitives
        self.pcd: Optional[o3d.geometry.PointCloud] = o3d.geometry.PointCloud()
        self.bbox: Optional[o3d.geometry.AxisAlignedBoundingBox] = (
            o3d.geometry.AxisAlignedBoundingBox()
        )

        # high level feats
        self.clip_ft: Optional[np.ndarray] = np.empty(0, dtype=np.float32)

        # class id
        self.class_id: Optional[int] = None

        # Initialize save_path
        self.save_path = self._initialize_save_path()

        # is navigation goal flag
        self.nav_goal = False

        # Debug / diagnostics for geometry update decisions.
        self.geometry_update_stats = Counter()
        self.last_geometry_update_mode = "init"

    def __getstate__(self):
        # Prepare the state dictionary for serialization
        state = {
            "uid": self.uid,
            "pcd_points": np.asarray(self.pcd.points).tolist(),  # Convert to list
            "pcd_colors": np.asarray(self.pcd.colors).tolist(),  # Convert to list
            "clip_ft": self.clip_ft.tolist(),
            "class_id": self.class_id,
            "nav_goal": self.nav_goal,
        }
        return state

    def __setstate__(self, state):
        self.uid = state.get("uid")

        # Restore PointCloud from points & colours.
        # self.pcd is always created to guarantee the attribute exists, even for
        # objects that lost all their points during merge + DBSCAN denoising.
        # Without this, any code accessing obj.pcd would raise AttributeError.

        points = np.array(state.get("pcd_points"))
        colors = np.array(state.get("pcd_colors"))

        self.pcd = o3d.geometry.PointCloud()
        if points.ndim == 2 and points.shape[1] == 3:
            self.pcd.points = o3d.utility.Vector3dVector(points)
        if colors.ndim == 2 and colors.shape[1] == 3:
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

        self.clip_ft = np.array(state.get("clip_ft"))
        self.class_id = state.get("class_id")
        self.nav_goal = state.get("nav_goal")

        self.observed_num = 0
        self.observations: List[str] = []

        self.save_path = self._initialize_save_path()
        self.geometry_update_stats = Counter()
        self.last_geometry_update_mode = "loaded"

    @classmethod
    def initialize_config(cls, config: DictConfig):
        cls._cfg = config

        classes_path = config.yolo.classes_path
        if config.yolo.use_given_classes:
            classes_path = config.yolo.given_classes_path
            logger.info(f"[BaseObject] Using given classes, path:{classes_path}")

        with open(classes_path, "r") as file:
            lines = file.readlines()
            num_classes = len(lines)
        # set num_classes for bayesian class filter
        cls._cfg.yolo.num_classes = num_classes

    def _initialize_save_path(self):
        if self._cfg:
            # save dir construction
            save_dir = self._cfg.map_save_path
            # If not exist, then create
            os.makedirs(save_dir, exist_ok=True)
            return os.path.join(save_dir, f"{self.uid}.pkl")

        return None

    def copy(self):
        return copy.deepcopy(self)

    def save_to_disk(self):
        """Save the object to disk using pickle."""
        with open(self.save_path, "wb") as f:
            pickle.dump(self, f)

        if self._cfg.save_cropped:
            # save the cropped image in the observation
            save_dir = self._cfg.map_save_path
            save_dir = os.path.join(save_dir, f"{self.class_id}_{self.uid}")
            cropped_save_dir = os.path.join(save_dir, "cropped")
            masked_save_dir = os.path.join(save_dir, "masked")
            os.makedirs(cropped_save_dir, exist_ok=True)
            os.makedirs(masked_save_dir, exist_ok=True)
            for obs in self.observations:
                obs_idx = obs.idx
                cropped_image = obs.cropped_image
                masked_image = obs.masked_image
                cropped_image_dir = os.path.join(cropped_save_dir, f"{obs_idx}.png")
                masked_image_dir = os.path.join(masked_save_dir, f"{obs_idx}.png")
                # both cropped and masked images are np.ndarray, so save as png
                import imageio

                imageio.imwrite(cropped_image_dir, cropped_image)
                imageio.imwrite(masked_image_dir, masked_image)

    @staticmethod
    def load_from_disk(filename: str):
        """Load the object from disk using pickle."""
        with open(filename, "rb") as f:
            obj = pickle.load(f)
            return obj

    def voxel_downsample_2d(
        self,
        pcd: o3d.geometry.PointCloud,
        voxel_size: float,
    ) -> o3d.geometry.PointCloud:
        # Input 3d point cloud, voxel size, return downsampled 2d point cloud
        # This function is to avoid the warning caused by o3d.geometry.voxel_down_sample
        # TODO: Color is not right

        if pcd is None or len(pcd.points) == 0:
            return o3d.geometry.PointCloud()

        # Get point cloud's points
        points_arr = np.asarray(pcd.points)
        colors_arr = np.asarray(pcd.colors)
        if points_arr.ndim != 2 or points_arr.shape[1] != 3:
            return o3d.geometry.PointCloud()
        if colors_arr.ndim != 2 or colors_arr.shape[1] != 3:
            colors_arr = np.zeros((len(points_arr), 3), dtype=np.float64)

        # Only retain X and Y coordinates
        points_2d = points_arr[:, :2]

        # 2D voxel downsample based on voxel size
        grid_indices = np.floor(points_2d / voxel_size).astype(np.int32)
        unique_indices, inverse_indices = np.unique(
            grid_indices, axis=0, return_inverse=True
        )

        # Calculate average points for each voxel
        downsampled_points_2d = np.zeros_like(unique_indices, dtype=np.float64)
        downsampled_colors = np.zeros((len(unique_indices), 3), dtype=np.float64)

        # calculate the mean of points in each voxel
        for i in range(len(unique_indices)):
            mask = inverse_indices == i
            downsampled_points_2d[i] = points_2d[mask].mean(axis=0)
            downsampled_colors[i] = colors_arr[mask].mean(axis=0)

        # restore the Z with given Z
        downsampled_points = np.zeros((len(downsampled_points_2d), 3))
        downsampled_points[:, :2] = downsampled_points_2d
        downsampled_points[:, 2] = self._cfg.floor_height

        # Generate the downsampled point cloud
        downsampled_pcd = o3d.geometry.PointCloud()
        downsampled_pcd.points = o3d.utility.Vector3dVector(downsampled_points)
        downsampled_pcd.colors = o3d.utility.Vector3dVector(downsampled_colors)

        return downsampled_pcd

    @staticmethod
    def copy_point_cloud(
        pcd: Optional[o3d.geometry.PointCloud],
    ) -> o3d.geometry.PointCloud:
        if pcd is None:
            return o3d.geometry.PointCloud()
        return copy.deepcopy(pcd)

    @staticmethod
    def point_count(pcd: Optional[o3d.geometry.PointCloud]) -> int:
        if pcd is None:
            return 0
        return len(pcd.points)

    def safe_get_axis_aligned_bounding_box(
        self, pcd: Optional[o3d.geometry.PointCloud]
    ) -> o3d.geometry.AxisAlignedBoundingBox:
        if pcd is None or len(pcd.points) == 0:
            return o3d.geometry.AxisAlignedBoundingBox()
        return pcd.get_axis_aligned_bounding_box()

    def get_object_geometry_update_mode(self) -> str:
        if self._cfg is None:
            return "append"

        configured_mode = getattr(self._cfg, "object_geometry_update_mode", None)
        if configured_mode is None or str(configured_mode).lower() == "auto":
            if getattr(self._cfg, "dataset_name", "") == "rosbag_tf":
                return "hybrid"
            return "append"

        mode = str(configured_mode).lower()
        if mode not in {"append", "replace", "hybrid"}:
            logger.warning(
                "[%s] Unknown object_geometry_update_mode='%s'. Falling back to append.",
                self.__class__.__name__,
                configured_mode,
            )
            return "append"
        return mode

    def get_geometry_update_thresholds(self) -> dict:
        legacy_voxel_size = float(
            getattr(
                self._cfg,
                "icp_voxel_size",
                getattr(self._cfg, "downsample_voxel_size", 0.01),
            )
        )
        legacy_distance_threshold = float(
            getattr(
                self._cfg,
                "icp_distance_threshold",
                max(legacy_voxel_size * 2.0, 0.02),
            )
        )
        return {
            "voxel_size": float(
                getattr(self._cfg, "object_geometry_icp_voxel_size", legacy_voxel_size)
            ),
            "distance_threshold": float(
                getattr(
                    self._cfg,
                    "object_geometry_icp_distance_threshold",
                    legacy_distance_threshold,
                )
            ),
            "min_points": int(
                getattr(
                    self._cfg,
                    "object_geometry_icp_min_points",
                    getattr(self._cfg, "icp_min_points", 20),
                )
            ),
            "min_fitness": float(
                getattr(
                    self._cfg,
                    "object_geometry_merge_min_fitness",
                    max(float(getattr(self._cfg, "icp_fitness_threshold", 0.3)), 0.85),
                )
            ),
            "max_rmse": float(
                getattr(
                    self._cfg,
                    "object_geometry_merge_max_rmse",
                    max(
                        float(getattr(self._cfg, "downsample_voxel_size", 0.01)),
                        legacy_distance_threshold * 0.5,
                    ),
                )
            ),
            "max_centroid_shift": float(
                getattr(
                    self._cfg,
                    "object_geometry_merge_max_centroid_shift",
                    max(
                        float(getattr(self._cfg, "downsample_voxel_size", 0.01)) * 1.5,
                        legacy_distance_threshold * 0.5,
                    ),
                )
            ),
        }

    def merge_clip_feature(self, incoming_clip_ft, previous_count=None) -> None:
        incoming_clip_ft = np.asarray(incoming_clip_ft, dtype=np.float32)
        if incoming_clip_ft.size == 0:
            return

        if previous_count is None:
            previous_count = max(self.observed_num - 1, 1)
        previous_count = max(int(previous_count), 1)

        if self.clip_ft.size == 0:
            self.clip_ft = incoming_clip_ft.copy()
        else:
            self.clip_ft = (
                (self.clip_ft * previous_count) + incoming_clip_ft
            ) / float(previous_count + 1)

        norm = np.linalg.norm(self.clip_ft)
        if norm > 1e-8:
            self.clip_ft = self.clip_ft / norm

    def denoise_point_cloud_dbscan(
        self, pcd: o3d.geometry.PointCloud
    ) -> o3d.geometry.PointCloud:
        if pcd is None or len(pcd.points) == 0:
            return o3d.geometry.PointCloud()

        eps = float(getattr(self._cfg, "dbscan_eps", 0.02))
        min_points = int(getattr(self._cfg, "dbscan_min_points", 10))
        try:
            pcd_clusters = np.array(
                pcd.cluster_dbscan(
                    eps=eps,
                    min_points=min_points,
                )
            )
        except RuntimeError:
            return pcd

        obj_points = np.asarray(pcd.points)
        obj_colors = np.asarray(pcd.colors)
        counter = Counter(pcd_clusters.tolist())
        if -1 in counter:
            del counter[-1]

        if not counter:
            return pcd

        most_common_label, _ = counter.most_common(1)[0]
        largest_mask = pcd_clusters == most_common_label
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]

        if len(largest_cluster_points) < 5:
            return pcd

        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
        return largest_cluster_pcd

    def apply_legacy_append_geometry_update(
        self,
        current_pcd: o3d.geometry.PointCloud,
        latest_pcd: o3d.geometry.PointCloud,
    ) -> tuple[o3d.geometry.PointCloud, dict]:
        current_copy = self.copy_point_cloud(current_pcd)
        latest_copy = self.copy_point_cloud(latest_pcd)
        metrics = {
            "fitness": np.nan,
            "rmse": np.nan,
            "centroid_shift": np.nan,
            "valid": True,
            "reason": "append",
        }

        if (
            getattr(self._cfg, "use_icp_alignment", False)
            and self.point_count(current_copy) > 0
            and self.point_count(latest_copy) > 0
        ):
            try:
                source_down = latest_copy.voxel_down_sample(
                    voxel_size=float(getattr(self._cfg, "icp_voxel_size", 0.05))
                )
                target_down = current_copy.voxel_down_sample(
                    voxel_size=float(getattr(self._cfg, "icp_voxel_size", 0.05))
                )
                icp_min_points = int(getattr(self._cfg, "icp_min_points", 10))
                if (
                    self.point_count(source_down) > icp_min_points
                    and self.point_count(target_down) > icp_min_points
                ):
                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        source_down,
                        target_down,
                        float(getattr(self._cfg, "icp_distance_threshold", 0.1)),
                        np.eye(4),
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
                    )
                    metrics["fitness"] = float(reg_p2p.fitness)
                    metrics["rmse"] = float(reg_p2p.inlier_rmse)
                    icp_fitness_threshold = float(
                        getattr(self._cfg, "icp_fitness_threshold", 0.3)
                    )
                    if reg_p2p.fitness > icp_fitness_threshold:
                        latest_copy.transform(reg_p2p.transformation)
                        metrics["reason"] = "append_icp_aligned"
                    else:
                        metrics["reason"] = "append_icp_rejected"
            except Exception as e:
                logger.warning(f"[{self.__class__.__name__}] Legacy ICP alignment failed: {e}")
                metrics["reason"] = "append_icp_failed"

        merged_pcd = current_copy
        merged_pcd += latest_copy
        return merged_pcd, metrics

    def build_geometry_update_candidate(
        self,
        current_pcd: o3d.geometry.PointCloud,
        latest_pcd: o3d.geometry.PointCloud,
    ) -> tuple[o3d.geometry.PointCloud, dict]:
        latest_copy = self.copy_point_cloud(latest_pcd)
        current_copy = self.copy_point_cloud(current_pcd)
        params = self.get_geometry_update_thresholds()
        metrics = {
            "fitness": 0.0,
            "rmse": float("inf"),
            "centroid_shift": float("inf"),
            "valid": False,
            "reason": "insufficient_points",
        }

        if self.point_count(current_copy) == 0 or self.point_count(latest_copy) == 0:
            return latest_copy, metrics

        source_down = current_copy.voxel_down_sample(params["voxel_size"])
        target_down = latest_copy.voxel_down_sample(params["voxel_size"])
        if (
            self.point_count(source_down) < params["min_points"]
            or self.point_count(target_down) < params["min_points"]
        ):
            return latest_copy, metrics

        try:
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_down,
                target_down,
                params["distance_threshold"],
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
            )
        except Exception as e:
            logger.warning(
                "[%s] Hybrid geometry ICP failed: %s",
                self.__class__.__name__,
                e,
            )
            metrics["reason"] = "icp_failed"
            return latest_copy, metrics

        aligned_current = self.copy_point_cloud(current_copy)
        aligned_current.transform(reg_p2p.transformation)
        current_center = np.asarray(
            self.safe_get_axis_aligned_bounding_box(aligned_current).get_center(),
            dtype=np.float32,
        )
        latest_center = np.asarray(
            self.safe_get_axis_aligned_bounding_box(latest_copy).get_center(),
            dtype=np.float32,
        )
        centroid_shift = float(np.linalg.norm(current_center - latest_center))

        metrics.update(
            {
                "fitness": float(reg_p2p.fitness),
                "rmse": float(reg_p2p.inlier_rmse),
                "centroid_shift": centroid_shift,
            }
        )
        metrics["valid"] = (
            metrics["fitness"] >= params["min_fitness"]
            and metrics["rmse"] <= params["max_rmse"]
            and metrics["centroid_shift"] <= params["max_centroid_shift"]
        )
        metrics["reason"] = (
            "accepted" if metrics["valid"] else "quality_gate_failed"
        )

        merged_pcd = aligned_current
        merged_pcd += latest_copy
        if self.point_count(merged_pcd) > 0:
            merged_pcd = merged_pcd.voxel_down_sample(
                voxel_size=float(getattr(self._cfg, "downsample_voxel_size", 0.01))
            )

        return merged_pcd, metrics

    def apply_geometry_update(
        self,
        current_pcd: o3d.geometry.PointCloud,
        latest_pcd: o3d.geometry.PointCloud,
        *,
        log_decision: bool = False,
    ) -> tuple[o3d.geometry.PointCloud, str, dict]:
        mode = self.get_object_geometry_update_mode()

        if mode == "replace":
            updated_pcd = self.copy_point_cloud(latest_pcd)
            mode_used = "replaced"
            metrics = {
                "fitness": np.nan,
                "rmse": np.nan,
                "centroid_shift": 0.0,
                "valid": True,
                "reason": "replace",
            }
        elif mode == "hybrid":
            merged_candidate, metrics = self.build_geometry_update_candidate(
                current_pcd=current_pcd,
                latest_pcd=latest_pcd,
            )
            if metrics["valid"]:
                updated_pcd = merged_candidate
                mode_used = "merged"
            else:
                updated_pcd = self.copy_point_cloud(latest_pcd)
                mode_used = "replaced"
        else:
            updated_pcd, metrics = self.apply_legacy_append_geometry_update(
                current_pcd=current_pcd,
                latest_pcd=latest_pcd,
            )
            mode_used = "appended"

        self.geometry_update_stats[mode_used] += 1
        self.last_geometry_update_mode = mode_used

        if log_decision and mode == "hybrid":
            logger.info(
                "[%s] geometry_update=%s uid=%s fitness=%.3f rmse=%.4f centroid_shift=%.4f reason=%s",
                self.__class__.__name__,
                mode_used,
                self.uid,
                float(metrics.get("fitness", np.nan)),
                float(metrics.get("rmse", np.nan)),
                float(metrics.get("centroid_shift", np.nan)),
                metrics.get("reason", "unknown"),
            )

        return updated_pcd, mode_used, metrics


class LocalObject(BaseObject):

    # Global variable for overall idx
    _curr_idx = 0

    def __init__(self):
        super().__init__()

        # lm sign
        self.is_low_mobility: Optional[bool] = False
        # major plane info, the z value of the major plane
        self.major_plane_info = None

        # Split Check info dict
        self.split_info: Optional[dict] = {}
        self.max_common: int = 0
        self.should_split: Optional[bool] = False
        # debug for split feat
        self.split_class_id_one: Optional[int] = 0
        self.split_class_id_two: Optional[int] = 0

        # Spatial Stablity Check Info List
        self.spatial_stable_info: Optional[list] = []

        # status
        self.status = LocalObjStatus.UPDATING
        self.is_stable = False
        self.pending_count = 0
        self.waiting_count = 0

        # # bayesian stable
        self.num_classes = self._cfg.yolo.num_classes
        # # init the prob
        self.class_probs = np.ones(self.num_classes) / self.num_classes
        self.class_probs_history: List[str] = []
        self.max_prob = 0.0
        self.entropy = 0.0
        self.change_rate = 0.0

        # for local map merging
        self.is_merged = False

        ################
        # Debug Variable
        ################
        self.downsample_num: int = 0
        self.unknown_class_id = self.num_classes - 1

    @classmethod
    def set_curr_idx(cls, idx: int):
        cls._curr_idx = idx

    def is_local_split_enabled(self) -> bool:
        if self._cfg is None:
            return True
        configured_value = getattr(self._cfg, "enable_local_split", None)
        if configured_value is not None:
            return bool(configured_value)
        return getattr(self._cfg, "dataset_name", "") != "rosbag_tf"

    def add_observation(self, observation: Observation) -> None:
        self.observations.append(observation)
        self.observed_num += 1

    def get_latest_observation(self) -> Observation:
        return self.observations[-1] if self.observations else None

    def clear_info(
        self,
    ) -> None:
        self.observed_num = 0
        self.observations = []
        self.pcd = o3d.geometry.PointCloud()
        # self.bbox = o3d.geometry.OrientedBoundingBox()
        self.class_id = None
        self.split_info = None
        self.max_common = 0
        self.should_split = False
        self.split_class_id_one = None
        self.split_class_id_two = None
        self.spatial_stable_info = None

    def resolve_class_id_from_observations(self, previous_class_id=None):
        class_ids = [obs.class_id for obs in self.observations if obs.class_id is not None]
        if not class_ids:
            return previous_class_id

        known_ids = [
            class_id for class_id in class_ids if class_id != self.unknown_class_id
        ]
        if known_ids:
            counts = Counter(known_ids)
            max_count = max(counts.values())
            candidates = [
                class_id for class_id, count in counts.items() if count == max_count
            ]
            if previous_class_id in candidates and previous_class_id != self.unknown_class_id:
                return previous_class_id
            return sorted(candidates)[0]

        if previous_class_id is not None and previous_class_id != self.unknown_class_id:
            return previous_class_id

        counts = Counter(class_ids)
        max_count = max(counts.values())
        candidates = [
            class_id for class_id, count in counts.items() if count == max_count
        ]
        if previous_class_id in candidates and previous_class_id is not None:
            return previous_class_id
        return sorted(candidates)[0]

    # Baye: Update posterior probability
    def update_class_probs(
        self,
        alpha=0.6,
    ) -> None:

        latest_obs = self.get_latest_observation()

        distance = latest_obs.distance

        # Current observation's confidence and class
        class_id = latest_obs.class_id
        conf = latest_obs.conf

        # Create class probability distribution for observation
        observation_probs = np.zeros(self.num_classes)

        # Distribute remaining confidence evenly among other classes
        remaining_conf = 1 - conf

        # Find top 10 classes with highest scores excluding current class
        all_indices = np.arange(self.num_classes)
        other_indices = all_indices[
            all_indices != class_id
        ]  # Indices excluding current class
        top_10_indices = np.argsort(self.class_probs[other_indices])[
            -10:
        ]  # Get top 10 classes with highest scores
        top_10_real_indices = other_indices[top_10_indices]  # Get real class indices

        # Distribute remaining confidence evenly among these 3 classes
        for idx in top_10_real_indices:
            observation_probs[idx] = remaining_conf / 10

        # Confidence of the current observation class
        observation_probs[class_id] = conf
        observation_probs /= np.sum(observation_probs)

        # Sliding window smoothing
        window_size = 5

        self.class_probs_history.append(observation_probs)
        if len(self.class_probs_history) > window_size:
            self.class_probs_history.pop(0)

        smoothed_probs = np.mean(self.class_probs_history, axis=0)

        # calculate alpha
        max_distance = 10.0
        k = 1.0
        alpha = np.exp(-k * distance / max_distance)

        # Bayesian update
        if class_id != self.unknown_class_id:
            self.class_probs = (1 - alpha) * self.class_probs + alpha * smoothed_probs
        else:
            # If unknown, we don't update the Bayesian filter to avoid diluting specific labels
            # But we still keep history for status/stability if needed (optional)
            pass

        # Normalize
        self.class_probs /= np.sum(self.class_probs)

    def update_info(self) -> None:
        # Local Obj
        # Integrate information from observation into external
        # This is the update part in the filter
        # TODO: Check how many times is appropriate for point cloud downsampling

        latest_obs = self.get_latest_observation()

        if self.observed_num == 0:
            logger.error("[LocalObject] No observation in this object")
            return

        if self.observed_num == 1:
            self.pcd = self.copy_point_cloud(latest_obs.pcd)
            self.bbox = self.safe_get_axis_aligned_bounding_box(self.pcd)
            self.clip_ft = np.asarray(latest_obs.clip_ft, dtype=np.float32).copy()
            self.class_id = latest_obs.class_id
            self.is_low_mobility = latest_obs.is_low_mobility
            return

        # Split dict update
        if self.is_local_split_enabled():
            self.update_split_info(latest_obs)
            if self.should_split:
                return

        self.pcd, _, _ = self.apply_geometry_update(
            current_pcd=self.pcd,
            latest_pcd=latest_obs.pcd,
            log_decision=self.get_object_geometry_update_mode() == "hybrid",
        )

        # Get new bbox
        self.bbox = self.safe_get_axis_aligned_bounding_box(self.pcd)

        # Merge clip_ft
        self.merge_clip_feature(
            latest_obs.clip_ft,
            previous_count=max(self.observed_num - 1, 1),
        )

        # Update spatial stable info list
        self.update_spatial_stable_info(latest_obs)

        previous_class_id = self.class_id
        self.class_id = self.resolve_class_id_from_observations(previous_class_id)

        # Bayesian update
        self.update_class_probs()

        # Get low mobility info
        # get low mobility info list
        low_mobility_infos = [obs.is_low_mobility for obs in self.observations]
        num_true = sum(low_mobility_infos)
        num_false = len(low_mobility_infos) - num_true
        most_common_lm = True if num_true > num_false else False

        self.is_low_mobility = most_common_lm

        # Periodically downsample the pcd
        if self.observed_num % self._cfg.downsample_interval == 0:
            self.downsample_num += 1
            self.pcd = self.pcd.voxel_down_sample(
                voxel_size=self._cfg.downsample_voxel_size
            )

            # only after downsample, we calculate the major plane info
            # If the object is low mobility, do major plane info calculation
            # TODO: do not need to calculate major plane info from scratch, incrementally calculate z_value
            # The major plane info can be calculated in the Observation Generation
            # Merging using the point number as the weight process
            if self.is_low_mobility:
                self.major_plane_info = self.find_major_plane_info()
            else:
                self.major_plane_info = None

    def update_split_info(self, latest_obs: Observation) -> None:
        if not self.is_local_split_enabled():
            self.should_split = False
            self.max_common = 0
            self.split_class_id_one = 0
            self.split_class_id_two = 0
            return

        # https://www.yuque.com/u21262689/fxzc7g/fmfk1gkv6fbemlus?singleDoc#
        # Boundary condition check
        if latest_obs is None:
            raise ValueError("latest_obs cannot be None")
        if not hasattr(latest_obs, "class_id") or not hasattr(latest_obs, "idx"):
            raise ValueError("latest_obs must have class_id and idx attributes")

        # Split dict update
        if latest_obs.class_id not in self.split_info:
            self.split_info[latest_obs.class_id] = deque()
        self.split_info[latest_obs.class_id].append(latest_obs.idx)

        # Make sure the idx in the active window
        # Current window size is 10
        for class_id, idx_deque in self.split_info.items():
            while (
                idx_deque
                and idx_deque[0] <= latest_obs.idx - self._cfg.active_window_size
            ):
                idx_deque.popleft()
            # # if empty, delete
            # if len(idx_deque) == 0:
            #     del self.split_info[class_id]
            if len(idx_deque) < self._cfg.active_window_size:
                continue

        # logger.info("current idx: ", latest_obs.idx)
        # for class_id, idx_deque in self.split_info.items():
        #     logger.info(f"Class ID: {class_id}, Observations: {list(idx_deque)}")

        # TODO: WHAT about three deque? currently only two considered
        split_tuple = self.find_max_common_elements(self.split_info)
        # logger.info slipt tupe
        # logger.info("Max common elements:", split_tuple)
        self.max_common = split_tuple[0]
        if split_tuple[0] != 0:
            self.split_class_id_one = split_tuple[1][0]
            self.split_class_id_two = split_tuple[1][1]
        else:
            self.split_class_id_one = 0
            self.split_class_id_two = 0

        if self.max_common > self._cfg.max_common_th:
            self.should_split = True

    def print_split_info(self) -> None:
        # logger.info split info into a string
        for class_id, idx_deque in self.split_info.items():
            logger.info(
                f"[LocalObject] Class ID: {class_id}, Observations: {list(idx_deque)}"
            )

    def print_split_info(
        self,
    ) -> str:
        split_info = ""
        for class_id, idx_deque in self.split_info.items():
            string_split = f"{class_id}" + "-" + f"{list(idx_deque)}"
            split_info = split_info + string_split + "||"
        return split_info

    def find_max_common_elements(
        self,
        data: dict,
    ) -> tuple:
        # Find max common elements in split info

        max_common_count = 0
        max_common_pair = (None, None)

        class_ids = list(data.keys())

        for i in range(len(class_ids)):
            for j in range(i + 1, len(class_ids)):
                class_id1 = class_ids[i]
                class_id2 = class_ids[j]

                deque1 = data[class_id1]
                deque2 = data[class_id2]

                # jump the empty deque
                if not deque1 or not deque2:
                    continue

                set1 = set(deque1)
                set2 = set(deque2)

                common_elements = set1 & set2
                common_count = len(common_elements)

                if common_count > max_common_count:
                    max_common_count = common_count
                    max_common_pair = (class_id1, class_id2)

        max_common_tuple = (max_common_count, max_common_pair)

        return max_common_tuple

    def update_info_from_observations(
        self,
    ) -> None:
        # This function rebuilds the entire object state from its stored observations.
        # It is used by merge_local_object() after combining observations from multiple
        # duplicate objects into a single new LocalObject.
        # IMPORTANT: This must reconstruct ALL fields that __getstate__ serialises,
        # including pcd_2d and bbox_2d, which are required for correct deserialisation.
        if self.observed_num == 0:
            logger.error("[LocalObject] No observations available for reconstruction.")
            return

        self.pcd = o3d.geometry.PointCloud()
        self.bbox = o3d.geometry.AxisAlignedBoundingBox()
        self.clip_ft = np.empty(0, dtype=np.float32)
        self.class_id = None
        self.is_low_mobility = False
        self.major_plane_info = None
        self.geometry_update_stats = Counter()
        self.last_geometry_update_mode = "rebuild"

        if hasattr(self, "num_classes"):
            self.class_probs = np.ones(self.num_classes) / self.num_classes
            self.class_probs_history = []
            self.max_prob = 0.0
            self.entropy = 0.0
            self.change_rate = 0.0

        if hasattr(self, "pcd_2d"):
            self.pcd_2d = o3d.geometry.PointCloud()
            self.bbox_2d = o3d.geometry.AxisAlignedBoundingBox()

        for obs_idx, obs in enumerate(self.observations):
            if obs_idx == 0:
                self.pcd = self.copy_point_cloud(obs.pcd)
                self.bbox = self.safe_get_axis_aligned_bounding_box(self.pcd)
                self.clip_ft = np.asarray(obs.clip_ft, dtype=np.float32).copy()
                self.class_id = obs.class_id
                self.is_low_mobility = obs.is_low_mobility
                continue

            self.pcd, _, _ = self.apply_geometry_update(
                current_pcd=self.pcd,
                latest_pcd=obs.pcd,
                log_decision=False,
            )
            self.bbox = self.safe_get_axis_aligned_bounding_box(self.pcd)
            self.merge_clip_feature(obs.clip_ft, previous_count=obs_idx)

            if hasattr(self, "pcd_2d") and hasattr(obs, "pcd_2d") and len(obs.pcd_2d.points) > 0:
                if len(self.pcd_2d.points) == 0:
                    self.pcd_2d = self.copy_point_cloud(obs.pcd_2d)
                else:
                    self.pcd_2d += self.copy_point_cloud(obs.pcd_2d)

        if len(self.pcd.points) > 0:
            self.pcd = self.pcd.voxel_down_sample(
                voxel_size=self._cfg.downsample_voxel_size
            )
        if len(self.pcd.points) > 0:
            self.pcd = self.denoise_point_cloud_dbscan(self.pcd)

        self.bbox = self.safe_get_axis_aligned_bounding_box(self.pcd)

        # Rebuild pcd_2d from observations.
        # pcd_2d is the 2D projection point cloud used by GlobalObject for
        # bounding box visualisation and class-weighted merging. Without this,
        # merged objects would have an empty pcd_2d, causing a RuntimeError
        # when __setstate__ tries to deserialise the saved .pkl file.
        if hasattr(self, 'pcd_2d'):
            pcd_2d_counter = 0
            for obs in self.observations:
                if hasattr(obs, 'pcd_2d') and len(obs.pcd_2d.points) > 0:
                    pcd_2d_counter += 1
                    if pcd_2d_counter == 1:
                        self.pcd_2d = obs.pcd_2d
                    else:
                        self.pcd_2d += obs.pcd_2d
            if pcd_2d_counter > 0:
                self.pcd_2d = self.voxel_downsample_2d(
                    pcd=self.pcd_2d, voxel_size=self._cfg.downsample_voxel_size
                )
                self.bbox_2d = self.pcd_2d.get_axis_aligned_bounding_box()

        self.class_id = self.resolve_class_id_from_observations(self.class_id)

        # Get low mobility info
        # get low mobility info list
        low_mobility_infos = [obs.is_low_mobility for obs in self.observations]
        num_true = sum(low_mobility_infos)
        num_false = len(low_mobility_infos) - num_true
        most_common_lm = True if num_true > num_false else False

        self.is_low_mobility = most_common_lm

        # get major plane info
        if self.is_low_mobility and len(self.pcd.points) > 0:
            self.major_plane_info = self.find_major_plane_info()

    def update_spatial_stable_info(self, latest_obs: Observation) -> None:
        # Including two parts of the spatial check
        # 1. latest obs bbox( or other primitives ) compare with the overall bbox or pcd
        # 2. latest obs bbox with prev obs bbox
        pass

    def update_status(
        self,
    ) -> None:
        # Object life cycle
        # Updating the status of the object by stability and infos

        # last observation time
        last_obs = self.get_latest_observation()

        # 1. if the object is in inside the sliding window, status will be UPDATING
        # No matter what previous status is, the object will always be UPDATING if in the window
        if last_obs.idx <= self._curr_idx and last_obs.idx >= max(
            self._curr_idx - self._cfg.active_window_size, 0
        ):
            self.status = LocalObjStatus.UPDATING
            self.pending_count = 0
            self.waiting_count = 0
            return

        # 2. if the object is out of the sliding window, check the stability first
        # and now the object is out of the window
        # Once the object is set as stable, it will always be stable
        self.stability_check()

        # if not stable, pending for next update, set status to PENDING
        if self.is_stable == False:
            self.status = LocalObjStatus.PENDING
            self.pending_count += 1

            # if pending count is large, status set as ELIMINATION
            if self.pending_count > self._cfg.max_pending_count:
                self.status = LocalObjStatus.ELIMINATION
                return

            return

        # 3. if the object is stable, waiting first
        # and now the obj is set as stable
        self.status = LocalObjStatus.WAITING
        self.waiting_count += 1

        if self.waiting_count < self._cfg.max_pending_count:
            # still in the waiting status, just return
            return

        # if waiting enough, then judge next step by lm status
        if self.is_low_mobility:
            self.status = LocalObjStatus.LM_ELIMINATION
            return
        else:
            self.status = LocalObjStatus.HM_ELIMINATION
            return

    def stability_check(
        self,
    ) -> None:
        # Check if the object is stable
        # the only function change the is_stable flag
        # Enhance the ways of stability check, here we only use label check
        # The initial check here is very simple, then goes bayesian stability check

        # 1. obs num should be large
        if self.observed_num < self._cfg.stable_num:
            self.is_stable = False
            return

        # 2. if the largest label over 1/3 of the observed num, just set as stable
        # Use prioritised label for check
        if self.class_id != self.unknown_class_id:
            class_ids = [obs.class_id for obs in self.observations]
            most_common_count = class_ids.count(self.class_id)
            if most_common_count > self.observed_num / 3:
                self.is_stable = True
                return
        else:
            # If still unknown, use normal count logic
            class_ids = [obs.class_id for obs in self.observations]
            obj_class_id_counter = Counter(class_ids)
            most_common_class_id, most_common_count = obj_class_id_counter.most_common(1)[0]
            if most_common_count > self.observed_num / 3:
                self.is_stable = True
                return

        # 3. if the object is stable by the filter, then set as stable
        if self.is_class_converged():
            self.is_stable = True
            return

        self.is_stable = False

    def is_class_converged(
        self,
        entropy_threshold=0.2,
        prob_threshold=0.50,
        change_rate_threshold=0.2,
        window_size=3,
    ) -> bool:

        # 1. Major class probability check
        max_prob = np.max(self.class_probs)
        self.max_prob = max_prob
        if max_prob > prob_threshold:
            return True

        # 2. Entropy check
        entropy = -np.sum(
            self.class_probs * np.log(self.class_probs + 1e-10)
        )  # prevent log(0)
        self.entropy = entropy
        if entropy < entropy_threshold:
            return True

        # 3. Change rate check
        if len(self.class_probs_history) >= window_size:
            recent_probs = np.array(self.class_probs_history[-window_size:])
            change_rate = np.mean(np.abs(recent_probs[1:] - recent_probs[:-1]), axis=0)
            self.change_rate = np.max(change_rate)
            if np.max(change_rate) < change_rate_threshold:
                return True

        return False

    def find_major_plane_info(
        self,
        bin_size=0.02,
    ) -> float:
        # This function will find the major plane of the object
        # return the major plane z value
        # Get the pcd
        # get all z_zxis value
        z_axis = np.asarray(self.pcd.points)[:, 2]

        z_min = z_axis.min()
        z_max = z_axis.max()

        # Fallback for synthetic/flat point clouds (e.g. from Depth-Anything)
        if (z_max - z_min) < bin_size:
            return float(z_axis[0])

        # Get the bin count
        bin_edges = np.arange(z_min, z_max + bin_size, bin_size)

        # Histogram calculation
        hist, bin_edges = np.histogram(z_axis, bins=bin_edges)

        # Optional: save the histogram

        # Find the peak of the histogram
        peak_index = np.argmax(hist)

        # Get the major plane z value
        major_plane_z = (bin_edges[peak_index] + bin_edges[peak_index + 1]) / 2.0

        return major_plane_z


class GlobalObject(BaseObject):
    def __init__(self, observation=None):
        super().__init__()

        # Spatial primitives
        self.pcd_2d: Optional[o3d.geometry.PointCloud] = o3d.geometry.PointCloud()
        self.bbox_2d: Optional[o3d.geometry.AxisAlignedBoundingBox] = (
            o3d.geometry.AxisAlignedBoundingBox()
        )

        # Related objs
        # Current we "only save clip feats" <-- PAY Attention!
        self.related_objs: List[np.ndarray] = []

        # for visualization in rerun
        self.related_bbox = []
        self.related_color = []

        self.class_observation_counts = {}
        self.known_observation_count = 0
        self.unknown_observation_count = 0

        # If provide the LocalObject, then initialize the GlobalObject from it
        if observation is not None:
            self.init_from_global_obs(observation)

    def __getstate__(self):
        # Base Object getstate
        state = super().__getstate__()

        # serialize List[np.ndarray]
        state["related_objs"] = [arr.tolist() for arr in self.related_objs]

        # serialize pcd_2d
        state["pcd_2d_points"] = np.asarray(self.pcd_2d.points).tolist()
        state["pcd_2d_colors"] = np.asarray(self.pcd_2d.colors).tolist()

        # Serialize related_bbox (convert AxisAlignedBoundingBox to dict)
        state["related_bbox"] = [
            {
                "min_bound": bbox.get_min_bound().tolist(),
                "max_bound": bbox.get_max_bound().tolist(),
            }
            for bbox in self.related_bbox
        ]

        # Serialize related_color (class IDs list)
        state["related_color"] = self.related_color  # assuming it's a list of class IDs
        state["class_observation_counts"] = self.class_observation_counts
        state["known_observation_count"] = self.known_observation_count
        state["unknown_observation_count"] = self.unknown_observation_count

        return state

    def __setstate__(self, state):
        # Base Object setstate
        super().__setstate__(state)

        # Restore related_objs as np.ndarray
        self.related_objs = [np.array(arr) for arr in state.get("related_objs", [])]

        # Restore pcd_2d (points and colours).
        # When an object was merged via update_info_from_observations() and no
        # observations had a valid pcd_2d, the serialised arrays will be empty lists.
        # np.array([]) produces shape (0,) which Open3D's Vector3dVector cannot accept
        # (it requires (N, 3)). We guard against this by only assigning points/colours
        # when the array is non-empty and correctly shaped.
        raw_points = state.get("pcd_2d_points", [])
        raw_colors = state.get("pcd_2d_colors", [])

        self.pcd_2d = o3d.geometry.PointCloud()
        if len(raw_points) > 0:
            points = np.array(raw_points)
            colors = np.array(raw_colors)
            if points.ndim == 2 and points.shape[1] == 3:
                self.pcd_2d.points = o3d.utility.Vector3dVector(points)
            if colors.ndim == 2 and colors.shape[1] == 3:
                self.pcd_2d.colors = o3d.utility.Vector3dVector(colors)

        self.bbox_2d = self.pcd_2d.get_axis_aligned_bounding_box()

        self.related_bbox = [
            o3d.geometry.AxisAlignedBoundingBox(
                min_bound=np.array(bbox_dict["min_bound"]),
                max_bound=np.array(bbox_dict["max_bound"]),
            )
            for bbox_dict in state.get("related_bbox", [])
        ]

        # Restore related_color (assuming it's stored as a list of class IDs)
        self.related_color = state.get("related_color", [])
        self.class_observation_counts = {
            int(class_id): int(count)
            for class_id, count in state.get("class_observation_counts", {}).items()
        }
        self.known_observation_count = int(state.get("known_observation_count", 0))
        self.unknown_observation_count = int(state.get("unknown_observation_count", 0))

        if not self.class_observation_counts and self.class_id is not None:
            self.record_class_observation(self.class_id)

        # Set obs num to 1 to avoid global updating bug
        self.observed_num = 1

    def copy(self):
        return copy.deepcopy(self)

    def init_from_global_obs(self, observation):

        self.uid = observation.uid

        self.pcd = self.copy_point_cloud(observation.pcd)
        self.bbox = self.safe_get_axis_aligned_bounding_box(self.pcd)

        self.pcd_2d = self.copy_point_cloud(observation.pcd_2d)
        self.bbox_2d = self.safe_get_axis_aligned_bounding_box(self.pcd_2d)

        self.clip_ft = np.asarray(observation.clip_ft, dtype=np.float32).copy()

        # Class ID
        self.class_id = observation.class_id

        # Related objs
        self.related_objs = observation.related_objs
        self.record_class_observation(self.class_id)

    def add_observation(self, observation: Observation) -> None:
        self.observations.append(observation)
        self.observed_num += 1

    def get_latest_observation(self) -> Observation:
        return self.observations[-1] if self.observations else None

    def get_unknown_class_id(self):
        if self._cfg is None:
            return None
        return getattr(self._cfg, "unknown_class_id", None)

    def is_unknown_class_id(self, class_id) -> bool:
        unknown_class_id = self.get_unknown_class_id()
        return unknown_class_id is not None and class_id == unknown_class_id

    def record_class_observation(self, class_id) -> None:
        if class_id is None:
            return

        class_id = int(class_id)
        self.class_observation_counts[class_id] = (
            self.class_observation_counts.get(class_id, 0) + 1
        )
        if self.is_unknown_class_id(class_id):
            self.unknown_observation_count += 1
        else:
            self.known_observation_count += 1

    def resolve_class_id_from_history(self):
        if not self.class_observation_counts:
            return self.class_id

        unknown_class_id = self.get_unknown_class_id()
        known_counts = {
            class_id: count
            for class_id, count in self.class_observation_counts.items()
            if unknown_class_id is None or class_id != unknown_class_id
        }

        if known_counts:
            max_count = max(known_counts.values())
            candidates = [
                class_id
                for class_id, count in known_counts.items()
                if count == max_count
            ]
            if (
                self.class_id in candidates
                and self.class_id is not None
                and not self.is_unknown_class_id(self.class_id)
            ):
                return self.class_id
            return sorted(candidates)[0]

        if unknown_class_id is not None and unknown_class_id in self.class_observation_counts:
            return unknown_class_id

        max_count = max(self.class_observation_counts.values())
        candidates = [
            class_id
            for class_id, count in self.class_observation_counts.items()
            if count == max_count
        ]
        if self.class_id in candidates and self.class_id is not None:
            return self.class_id
        return sorted(candidates)[0]

    def merge_clip_feature(self, incoming_clip_ft, previous_count=None) -> None:
        super().merge_clip_feature(incoming_clip_ft, previous_count=previous_count)

    def update_info(self) -> None:
        # Global Obj

        latest_obs = self.get_latest_observation()

        if self.observed_num == 0:
            logger.error("[GlobalObject] No observation in this object")
            return

        if self.observed_num == 1:
            self.uid = latest_obs.uid

            self.pcd = self.copy_point_cloud(latest_obs.pcd)
            self.bbox = self.safe_get_axis_aligned_bounding_box(self.pcd)
            self.pcd_2d = self.copy_point_cloud(latest_obs.pcd_2d)
            self.bbox_2d = self.safe_get_axis_aligned_bounding_box(self.pcd_2d)

            self.clip_ft = np.asarray(latest_obs.clip_ft, dtype=np.float32).copy()
            self.class_id = latest_obs.class_id

            self.related_objs = copy.deepcopy(latest_obs.related_objs)

            # for visualization in rerun
            self.related_bbox = copy.deepcopy(latest_obs.related_bbox)
            self.related_color = copy.deepcopy(latest_obs.related_color)
            if not self.class_observation_counts:
                self.record_class_observation(self.class_id)
            return

        # Update the information for outside

        self.pcd, _, _ = self.apply_geometry_update(
            current_pcd=self.pcd,
            latest_pcd=latest_obs.pcd,
            log_decision=self.get_object_geometry_update_mode() == "hybrid",
        )

        # Get new bbox
        self.bbox = self.safe_get_axis_aligned_bounding_box(self.pcd)

        self.pcd_2d = self.voxel_downsample_2d(
            pcd=self.copy_point_cloud(self.pcd),
            voxel_size=self._cfg.downsample_voxel_size,
        )
        self.bbox_2d = self.safe_get_axis_aligned_bounding_box(self.pcd_2d)

        self.merge_clip_feature(
            latest_obs.clip_ft,
            previous_count=max(self.observed_num - 1, 1),
        )
        self.record_class_observation(latest_obs.class_id)
        self.class_id = self.resolve_class_id_from_history()

        # TODO: Any other matching strategy on related objs?
        # Maintain the related objs, simply add the objs from the latest observation (Current Strategy)
        self.related_objs += latest_obs.related_objs

        # for visualization in rerun
        self.related_bbox += latest_obs.related_bbox
        self.related_color += latest_obs.related_color

        pass
