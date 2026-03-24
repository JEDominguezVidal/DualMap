import logging
import pdb
from typing import List

import faiss
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from scipy.sparse.csgraph import connected_components

from utils.object import BaseObject
from utils.types import Observation
from utils.visualizer import plot_similarity_matrix

# Set up the module-level logger
logger = logging.getLogger(__name__)


class Tracker:
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        # Construct different tracker types based on cfg.tracker classification
        # config
        self.cfg = cfg

        self.__is_global = False

        self.merge_info = None
        self.last_added_new_objects = 0

    def set_ref_map(
        self,
        ref_map: List[BaseObject],
    ) -> None:
        self.ref_map = ref_map

    def set_ref_frame(
        self,
        ref_frame: List[Observation],
    ) -> None:
        self.ref_frame = ref_frame

    def set_current_frame(self, curr_frame) -> None:
        # TODO: Deepcopy?
        self.curr_frame = curr_frame

    def get_current_frame(
        self,
    ):
        return self.curr_frame

    def get_merge_info(self):
        return self.merge_info

    def get_last_added_new_objects(self):
        return self.last_added_new_objects

    def set_global(
        self,
    ) -> None:
        self.__is_global = True
        return

    def is_unknown_class_id(self, class_id) -> bool:
        unknown_class_id = getattr(self.cfg, "unknown_class_id", None)
        return unknown_class_id is not None and class_id == unknown_class_id

    @staticmethod
    def compute_bbox_volume(bbox) -> float:
        extent = np.asarray(bbox.get_extent(), dtype=np.float32)
        return float(np.prod(np.maximum(extent, 1e-6)))

    def apply_local_match_vetoes(
        self,
        spatial_sim_mat: torch.Tensor,
        visual_sim_mat: torch.Tensor,
    ):
        max_bbox_volume_ratio = float(
            getattr(
                self.cfg.object_tracking,
                "max_bbox_volume_ratio",
                float("inf"),
            )
        )

        spatial_sim_mat = spatial_sim_mat.clone()
        visual_sim_mat = visual_sim_mat.clone()
        veto_count = 0

        for map_idx, map_obj in enumerate(self.ref_map):
            map_volume = self.compute_bbox_volume(map_obj.bbox)

            for obs_idx, obs in enumerate(self.curr_frame):
                if getattr(obs, "non_trackable", False):
                    spatial_sim_mat[map_idx, obs_idx] = 0.0
                    visual_sim_mat[map_idx, obs_idx] = 0.0
                    veto_count += 1
                    continue

                spatial_value = float(spatial_sim_mat[map_idx, obs_idx])

                obs_volume = self.compute_bbox_volume(obs.bbox)
                volume_ratio = max(map_volume, obs_volume) / max(
                    min(map_volume, obs_volume),
                    1e-6,
                )
                if (
                    np.isfinite(max_bbox_volume_ratio)
                    and volume_ratio > max_bbox_volume_ratio
                    and spatial_value < 0.2
                ):
                    spatial_sim_mat[map_idx, obs_idx] = 0.0
                    visual_sim_mat[map_idx, obs_idx] = 0.0
                    veto_count += 1

        if veto_count > 0:
            logger.info(
                "[Tracker] Applied %d local match vetoes before association.",
                veto_count,
            )

        return spatial_sim_mat, visual_sim_mat

    def matching_map(
        self,
        is_map_only: bool = False,
    ) -> None:
        # match current frame to the map
        # Find relationships between current observations and previous map

        if self.__is_global:
            self.match_global_greedy()

        else:
            if is_map_only:
                spatial_sim_mat = self.compute_overlap_spatial_sim()

                graph = spatial_sim_mat > self.cfg.merge_sim_threshold
                n_components, component_labels = connected_components(graph)
                # get merge info
                self.merge_info = [
                    np.where(component_labels == i)[0] for i in range(n_components)
                ]

            # Spatial sim torch
            # M(map) x N(curr)
            spatial_sim_mat = self.compute_spatial_sim()
            # plot_similarity_matrix(spatial_sim_mat)

            # visual sim torch
            # M(map) x N(curr)
            visual_sim_mat = self.compute_visual_sim()
            # plot_similarity_matrix(visual_sim_mat)

            if (
                spatial_sim_mat is None
                or visual_sim_mat is None
                or spatial_sim_mat.numel() == 0
                or visual_sim_mat.numel() == 0
                or spatial_sim_mat.shape != visual_sim_mat.shape
            ):
                logger.warning(
                    "[Tracker] Empty or mismatched similarity matrices, skipping match."
                )
                return

            spatial_sim_mat, visual_sim_mat = self.apply_local_match_vetoes(
                spatial_sim_mat, visual_sim_mat
            )

            # overall sim
            sim_mat = spatial_sim_mat + visual_sim_mat
            # switch (map, curr) to (curr, map)
            sim_mat = sim_mat.T

            if sim_mat.shape[0] == 0 or sim_mat.shape[1] == 0:
                logger.warning(
                    "[Tracker] sim_mat is empty after transpose, skipping update."
                )

                return

            bbox_iou_mat, centroid_dist_mat = self.compute_local_iou_and_centroid_dist()
            self.update_obs_with_sim_mat(
                sim_mat,
                spatial_sim_mat=spatial_sim_mat,
                visual_sim_mat=visual_sim_mat,
                bbox_iou_mat=bbox_iou_mat,
                centroid_dist_mat=centroid_dist_mat,
            )

    def compute_overlap_spatial_sim(self) -> np.ndarray:
        len_map = len(self.ref_map)
        len_curr = len(self.curr_frame)

        overlap_matrix = np.zeros((len_map, len_curr))

        # calculate iou first

        # Get stacked bboxes for iou calculation
        # from map
        map_bbox_values = []
        for obj in self.ref_map:
            obj_bbox = np.asarray(obj.bbox.get_box_points())
            obj_bbox = torch.from_numpy(obj_bbox)
            map_bbox_values.append(obj_bbox)
            
        if not map_bbox_values:
            return overlap_matrix
            
        map_bbox_torch = torch.stack(map_bbox_values, dim=0)

        # from curr obs
        curr_bbox_values = []
        for obj in self.curr_frame:
            obj_bbox = np.asarray(obj.bbox.get_box_points())
            obj_bbox = torch.from_numpy(obj_bbox)
            curr_bbox_values.append(obj_bbox)
            
        if not curr_bbox_values:
            return overlap_matrix
            
        curr_bbox_torch = torch.stack(curr_bbox_values, dim=0)

        # calculate iou
        iou = self.compute_3d_iou_batch(map_bbox_torch, curr_bbox_torch)

        # calculate centroids for fallback
        map_centroids = torch.mean(map_bbox_torch, dim=1) # (M, 3)
        curr_centroids = torch.mean(curr_bbox_torch, dim=1) # (N, 3)

        for idx_a in range(len_map):
            for idx_b in range(idx_a + 1, len_curr):
                centroid_dist = torch.norm(map_centroids[idx_a] - curr_centroids[idx_b]).item()

                if iou[idx_a, idx_b] < 1e-6 and centroid_dist > self.cfg.merging_centroid_dist:
                    continue

                pcd_map = self.ref_map[idx_a].pcd
                pcd_curr = self.curr_frame[idx_b].pcd

                overlap_matrix[idx_a, idx_b] = self.find_overlapping_ratio_faiss(
                    pcd_map, pcd_curr, radius=(self.cfg.faiss_radius_factor * self.cfg.downsample_voxel_size)
                )

        return overlap_matrix

    def compute_spatial_sim(self) -> np.ndarray:

        len_map = len(self.ref_map)
        len_curr = len(self.curr_frame)

        if len_map == 0 or len_curr == 0:
            return torch.zeros((len_map, len_curr))  # shape = (0, N) or (M, 0)

        overlap_matrix = np.zeros((len_map, len_curr))

        points_map = [
            np.asarray(obj.pcd.points, dtype=np.float32) for obj in self.ref_map
        ]  # m size
        indices_map = [
            faiss.IndexFlatL2(points_arr.shape[1]) for points_arr in points_map
        ]  # m indices
        for idx, points_arr in zip(indices_map, points_map):
            idx.add(points_arr)

        points_curr = [
            np.asarray(obs.pcd.points, dtype=np.float32) for obs in self.curr_frame
        ]

        # Get stacked bboxes for iou calculation
        # from map
        map_bbox_values = []
        for obj in self.ref_map:
            obj_bbox = np.asarray(obj.bbox.get_box_points())
            obj_bbox = torch.from_numpy(obj_bbox)
            map_bbox_values.append(obj_bbox)
            
        if not map_bbox_values:
            return torch.from_numpy(overlap_matrix)
            
        map_bbox_torch = torch.stack(map_bbox_values, dim=0)

        # from curr obs
        curr_bbox_values = []
        for obs in self.curr_frame:
            obs_bbox = np.asarray(obs.bbox.get_box_points())
            obs_bbox = torch.from_numpy(obs_bbox)
            curr_bbox_values.append(obs_bbox)
            
        if not curr_bbox_values:
            return torch.from_numpy(overlap_matrix)
            
        curr_bbox_torch = torch.stack(curr_bbox_values, dim=0)

        # calculate iou
        iou = self.compute_3d_iou_batch(map_bbox_torch, curr_bbox_torch)

        # calculate centroids for fallback
        map_centroids = torch.mean(map_bbox_torch, dim=1) # (M, 3)
        curr_centroids = torch.mean(curr_bbox_torch, dim=1) # (N, 3)

        counter = 0

        # compute the overlap info using pcd
        for idx_a in range(len_map):
            for idx_b in range(len_curr):

                centroid_dist = torch.norm(map_centroids[idx_a] - curr_centroids[idx_b]).item()

                if iou[idx_a, idx_b] < 1e-6 and centroid_dist > self.cfg.tracking_centroid_dist:
                    counter += 1
                    continue

                # Use configurable FAISS radius factor
                search_radius = self.cfg.faiss_radius_factor * self.cfg.downsample_voxel_size
                D, I = indices_map[idx_a].search(points_curr[idx_b], 1)
                overlap = (D < search_radius**2).sum()
                # calculate the ratio of points within the threshold distance
                denom = len(points_curr[idx_b])
                if denom == 0:
                    overlap_matrix[idx_a, idx_b] = 0.0
                else:
                    overlap_matrix[idx_a, idx_b] = overlap / denom

        overlap_matrix = torch.from_numpy(overlap_matrix)

        return overlap_matrix

    def compute_global_spatial_sim(
        self,
    ) -> np.ndarray:
        # compute globally spatial sim
        # we only use 2d bbox to judge here
        len_map = len(self.ref_map)
        len_curr = len(self.curr_frame)

        # Get stacked bboxes for iou calculation
        map_bbox_values = []

        for obj in self.ref_map:
            min_bound = obj.bbox_2d.get_min_bound()
            max_bound = obj.bbox_2d.get_max_bound()
            map_bbox_values.append(
                torch.tensor([min_bound[0], min_bound[1], max_bound[0], max_bound[1]])
            )
        
        if not map_bbox_values:
            return torch.zeros((0, len_curr))
            
        map_bbox_torch = torch.stack(map_bbox_values, dim=0)

        # from curr obs
        curr_bbox_values = []
        for obs in self.curr_frame:
            min_bound = obs.bbox_2d.get_min_bound()
            max_bound = obs.bbox_2d.get_max_bound()
            curr_bbox_values.append(
                torch.tensor([min_bound[0], min_bound[1], max_bound[0], max_bound[1]])
            )
            
        if not curr_bbox_values:
            return torch.zeros((len_map, 0))
            
        curr_bbox_torch = torch.stack(curr_bbox_values, dim=0)

        ratio = self.compute_match_by_intersection_ratio(
            map_bbox_torch, curr_bbox_torch
        )

        return ratio

    def compute_global_centroid_distances(self) -> np.ndarray:
        len_map = len(self.ref_map)
        len_curr = len(self.curr_frame)

        if len_map == 0 or len_curr == 0:
            return np.zeros((len_curr, len_map), dtype=np.float32)

        map_centers = np.array(
            [obj.bbox_2d.get_center()[:2] for obj in self.ref_map],
            dtype=np.float32,
        )
        curr_centers = np.array(
            [obs.bbox_2d.get_center()[:2] for obs in self.curr_frame],
            dtype=np.float32,
        )

        return np.linalg.norm(
            curr_centers[:, None, :] - map_centers[None, :, :],
            axis=2,
        ).astype(np.float32)

    def compute_local_iou_and_centroid_dist(self):
        len_map = len(self.ref_map)
        len_curr = len(self.curr_frame)

        if len_map == 0 or len_curr == 0:
            return (
                torch.zeros((len_map, len_curr), dtype=torch.float32),
                torch.zeros((len_map, len_curr), dtype=torch.float32),
            )

        map_bbox_values = []
        for obj in self.ref_map:
            obj_bbox = np.asarray(obj.bbox.get_box_points())
            map_bbox_values.append(torch.from_numpy(obj_bbox))

        curr_bbox_values = []
        for obs in self.curr_frame:
            obs_bbox = np.asarray(obs.bbox.get_box_points())
            curr_bbox_values.append(torch.from_numpy(obs_bbox))

        map_bbox_torch = torch.stack(map_bbox_values, dim=0)
        curr_bbox_torch = torch.stack(curr_bbox_values, dim=0)

        iou = self.compute_3d_iou_batch(map_bbox_torch, curr_bbox_torch)

        map_centroids = torch.mean(map_bbox_torch, dim=1).float()
        curr_centroids = torch.mean(curr_bbox_torch, dim=1).float()
        centroid_dist = torch.cdist(map_centroids, curr_centroids, p=2)

        return iou, centroid_dist.cpu()

    def compute_match_by_intersection_ratio(
        self, bboxes1: torch.Tensor, bboxes2: torch.Tensor, threshold=0.8
    ) -> torch.Tensor:
        """
        Calculate match matrix based on the intersection ratio between bounding boxes.

        bboxes1: torch.Tensor of shape (N, 4), first set of bounding boxes (min_x, min_y, max_x, max_y)
        bboxes2: torch.Tensor of shape (M, 4), second set of bounding boxes (min_x, min_y, max_x, max_y)
        threshold: match threshold, default is 0.8

        Returns: torch.Tensor of shape (N, M), match matrix
        """
        # Extract coordinates of bounding boxes
        bboxes1_min_x, bboxes1_min_y, bboxes1_max_x, bboxes1_max_y = (
            bboxes1[:, 0],
            bboxes1[:, 1],
            bboxes1[:, 2],
            bboxes1[:, 3],
        )
        bboxes2_min_x, bboxes2_min_y, bboxes2_max_x, bboxes2_max_y = (
            bboxes2[:, 0],
            bboxes2[:, 1],
            bboxes2[:, 2],
            bboxes2[:, 3],
        )

        # Compute intersection coordinates
        inter_min_x = torch.max(
            bboxes1_min_x[:, None], bboxes2_min_x
        )  # top-left x of intersection
        inter_min_y = torch.max(
            bboxes1_min_y[:, None], bboxes2_min_y
        )  # top-left y of intersection
        inter_max_x = torch.min(
            bboxes1_max_x[:, None], bboxes2_max_x
        )  # bottom-right x of intersection
        inter_max_y = torch.min(
            bboxes1_max_y[:, None], bboxes2_max_y
        )  # bottom-right y of intersection

        # Calculate intersection width and height, ensuring non-negative values
        inter_width = (inter_max_x - inter_min_x).clamp(min=0)
        inter_height = (inter_max_y - inter_min_y).clamp(min=0)

        # Calculate intersection area
        inter_area = inter_width * inter_height

        # Calculate the area of each bounding box
        bboxes1_area = (bboxes1_max_x - bboxes1_min_x) * (bboxes1_max_y - bboxes1_min_y)
        bboxes2_area = (bboxes2_max_x - bboxes2_min_x) * (bboxes2_max_y - bboxes2_min_y)

        # Calculate the ratio of intersection area to each bounding box's area
        ratio1 = inter_area / bboxes1_area[:, None]  # ratio for bboxes1
        ratio2 = inter_area / bboxes2_area  # ratio for bboxes2

        # Determine matches if either ratio exceeds the threshold
        # match_matrix = (ratio1 >= threshold) | (ratio2 >= threshold)
        match_matrix = torch.max(ratio1, ratio2)

        return match_matrix

    def compute_visual_sim(
        self,
    ) -> np.ndarray:
        # Get stacked clip fts for calculation
        # from map
        map_feats_values = []
        for obj in self.ref_map:
            obj_feat = torch.from_numpy(obj.clip_ft)
            map_feats_values.append(obj_feat)

        if not map_feats_values:
            return torch.zeros((0, len(self.curr_frame)))

        map_feats_torch = torch.stack(map_feats_values, dim=0)  # (M, D)

        # from curr obs
        curr_feats_values = []
        for obs in self.curr_frame:
            obs_feat = torch.from_numpy(obs.clip_ft)
            curr_feats_values.append(obs_feat)
            
        if not curr_feats_values:
            return torch.zeros((len(self.ref_map), 0))
            
        curr_feats_torch = torch.stack(curr_feats_values, dim=0)  # (N, D)

        map_fts = map_feats_torch.unsqueeze(-1)  # (M, D, 1)
        curr_fts = curr_feats_torch.T.unsqueeze(0)  # (1, D, N)

        visual_sim = F.cosine_similarity(map_fts, curr_fts, dim=1)  # (M, N)

        return visual_sim

    def update_obs_with_sim_mat(
        self,
        sim_mat: torch.Tensor,
        spatial_sim_mat: torch.Tensor | None = None,
        visual_sim_mat: torch.Tensor | None = None,
        bbox_iou_mat: torch.Tensor | None = None,
        centroid_dist_mat: torch.Tensor | None = None,
    ) -> None:
        # update the obs in current frame with the matched map
        # IF no matches, then the current obs matched places will be None

        # get len of the curr obs
        len_curr_obs = len(self.curr_frame)

        add_new_obj = 0
        spawn_reasons = {
            "volume_veto": 0,
            "low_clip": 0,
            "low_geometry": 0,
            "no_candidate": 0,
            "non_trackable": 0,
        }
        use_one_to_one = bool(getattr(self.cfg, "local_one_to_one_matching", True))
        assigned_map_indices = set()
        sim_mat_np = self.to_numpy_array(sim_mat)

        for obs in self.curr_frame:
            obs.matched_obj_idx = -1
            obs.matched_obj_uid = None
            obs.matched_obj_score = 0.0

        if use_one_to_one and sim_mat_np is not None and sim_mat_np.size > 0:
            candidates = []
            for obs_idx in range(len_curr_obs):
                if getattr(self.curr_frame[obs_idx], "non_trackable", False):
                    continue
                row = sim_mat_np[obs_idx]
                if row.size == 0:
                    continue
                map_idx = int(np.argmax(row))
                score = float(row[map_idx])
                if score > self.cfg.sim_threshold:
                    candidates.append((score, obs_idx, map_idx))

            for score, obs_idx, map_idx in sorted(
                candidates,
                key=lambda item: item[0],
                reverse=True,
            ):
                if map_idx in assigned_map_indices:
                    continue
                self.curr_frame[obs_idx].matched_obj_uid = self.ref_map[map_idx].uid
                self.curr_frame[obs_idx].matched_obj_score = score
                self.curr_frame[obs_idx].matched_obj_idx = map_idx
                assigned_map_indices.add(map_idx)

        # update information into current observation
        for idx in range(len_curr_obs):
            if self.curr_frame[idx].matched_obj_idx != -1:
                continue
            matched, reason = self.try_local_continuity_match(
                obs_idx=idx,
                spatial_sim_mat=spatial_sim_mat,
                visual_sim_mat=visual_sim_mat,
                bbox_iou_mat=bbox_iou_mat,
                centroid_dist_mat=centroid_dist_mat,
                excluded_map_indices=assigned_map_indices if use_one_to_one else None,
            )
            if matched:
                assigned_map_indices.add(self.curr_frame[idx].matched_obj_idx)
                continue
            add_new_obj += 1
            spawn_reasons[reason] = spawn_reasons.get(reason, 0) + 1

        self.last_added_new_objects = add_new_obj
        logger.info(
            f"[Tracker] Added {add_new_obj} new objects, current detections: {len_curr_obs}"
        )
        if add_new_obj > 0:
            logger.info("[Tracker] Spawn reasons for unmatched detections: %s", spawn_reasons)

    @staticmethod
    def to_numpy_array(matrix):
        if matrix is None:
            return None
        if isinstance(matrix, np.ndarray):
            return matrix
        if hasattr(matrix, "cpu") and hasattr(matrix, "numpy"):
            return matrix.cpu().numpy()
        return np.asarray(matrix)

    def try_local_continuity_match(
        self,
        obs_idx: int,
        spatial_sim_mat: torch.Tensor | None,
        visual_sim_mat: torch.Tensor | None,
        bbox_iou_mat: torch.Tensor | None,
        centroid_dist_mat: torch.Tensor | None,
        excluded_map_indices: set | None = None,
    ) -> tuple[bool, str]:
        if (
            spatial_sim_mat is None
            or visual_sim_mat is None
            or bbox_iou_mat is None
            or centroid_dist_mat is None
            or (spatial_sim_mat.numel() if hasattr(spatial_sim_mat, "numel") else np.size(spatial_sim_mat)) == 0
        ):
            return False, "no_candidate"

        obs = self.curr_frame[obs_idx]
        if getattr(obs, "non_trackable", False):
            return False, "non_trackable"

        min_clip_similarity = 0.30
        max_centroid_dist = 0.04
        min_overlap = 0.10
        min_bbox_iou = 0.02
        max_bbox_volume_ratio = float(
            getattr(
                self.cfg.object_tracking,
                "max_bbox_volume_ratio",
                float("inf"),
            )
        )

        best_candidate = None
        best_score = -np.inf
        best_reason = "no_candidate"

        for map_idx, map_obj in enumerate(self.ref_map):
            if excluded_map_indices is not None and map_idx in excluded_map_indices:
                continue
            bbox_iou = float(bbox_iou_mat[map_idx, obs_idx])
            spatial_overlap = float(spatial_sim_mat[map_idx, obs_idx])
            map_volume = self.compute_bbox_volume(map_obj.bbox)
            obs_volume = self.compute_bbox_volume(obs.bbox)
            volume_ratio = max(map_volume, obs_volume) / max(
                min(map_volume, obs_volume),
                1e-6,
            )
            if volume_ratio > max_bbox_volume_ratio and max(spatial_overlap, bbox_iou) < min_overlap:
                best_reason = "volume_veto"
                continue

            clip_cos = float(visual_sim_mat[map_idx, obs_idx])
            if clip_cos < min_clip_similarity:
                if best_reason == "no_candidate":
                    best_reason = "low_clip"
                continue

            centroid_dist = float(centroid_dist_mat[map_idx, obs_idx])
            if centroid_dist > max_centroid_dist:
                if best_reason in {"no_candidate", "low_clip"}:
                    best_reason = "low_geometry"
                continue

            centroid_score = max(
                0.0, 1.0 - (centroid_dist / max(max_centroid_dist, 1e-6))
            )
            score = clip_cos + max(spatial_overlap, bbox_iou) + (0.25 * centroid_score)
            if score > best_score:
                best_score = score
                best_candidate = map_idx

        if best_candidate is None:
            return False, best_reason

        self.curr_frame[obs_idx].matched_obj_uid = self.ref_map[best_candidate].uid
        self.curr_frame[obs_idx].matched_obj_score = float(best_score)
        self.curr_frame[obs_idx].matched_obj_idx = best_candidate
        return True, "matched"

    @staticmethod
    def compute_centroid_score(distance: float, max_distance: float) -> float:
        return max(0.0, 1.0 - (distance / max(max_distance, 1e-6)))

    def get_global_match_spec(self, obs_class, map_class):
        obs_unknown = self.is_unknown_class_id(obs_class)
        map_unknown = self.is_unknown_class_id(map_class)

        if not obs_unknown and not map_unknown:
            if obs_class != map_class:
                return None
            return {
                "name": "same_known",
                "min_clip": 0.25,
                "max_centroid": 0.04,
                "min_overlap": 0.20,
                "min_score": 0.55,
            }

        if obs_unknown and not map_unknown:
            return {
                "name": "unknown_to_known",
                "min_clip": 0.35,
                "max_centroid": 0.03,
                "min_overlap": 0.35,
                "min_score": 0.65,
            }

        if obs_unknown and map_unknown:
            return {
                "name": "unknown_to_unknown",
                "min_clip": 0.35,
                "max_centroid": 0.03,
                "min_overlap": 0.35,
                "min_score": 0.65,
            }

        return None

    def match_global_greedy(self) -> None:
        len_curr = len(self.curr_frame)
        len_map = len(self.ref_map)

        for obs in self.curr_frame:
            obs.matched_obj_idx = -1
            obs.matched_obj_uid = None
            obs.matched_obj_score = 0.0

        if len_curr == 0 or len_map == 0:
            self.last_added_new_objects = len_curr
            logger.info(
                "[Tracker][Global] Added %d new objects, current observations: %d",
                len_curr,
                len_curr,
            )
            return

        overlap_mat = self.compute_global_spatial_sim().T.cpu().numpy()
        centroid_dist_mat = self.compute_global_centroid_distances()
        clip_sim_mat = self.compute_visual_sim().T.cpu().numpy()
        candidates = []
        for obs_idx, obs in enumerate(self.curr_frame):
            obs_class = getattr(obs, "class_id", None)
            for map_idx, map_obj in enumerate(self.ref_map):
                map_class = getattr(map_obj, "class_id", None)
                spec = self.get_global_match_spec(obs_class, map_class)
                if spec is None:
                    continue

                clip_cos = float(clip_sim_mat[obs_idx, map_idx])
                if clip_cos < spec["min_clip"]:
                    continue

                overlap = float(overlap_mat[obs_idx, map_idx])
                centroid_dist = float(centroid_dist_mat[obs_idx, map_idx])
                if overlap < spec["min_overlap"] and centroid_dist > spec["max_centroid"]:
                    continue

                centroid_score = self.compute_centroid_score(
                    centroid_dist, spec["max_centroid"]
                )
                score = (0.50 * clip_cos) + (0.35 * overlap) + (0.15 * centroid_score)
                if score < spec["min_score"]:
                    continue

                candidates.append((score, obs_idx, map_idx))

        assigned_obs = set()
        assigned_map = set()
        match_count = 0
        for score, obs_idx, map_idx in sorted(candidates, key=lambda item: item[0], reverse=True):
            if obs_idx in assigned_obs or map_idx in assigned_map:
                continue
            self.curr_frame[obs_idx].matched_obj_uid = self.ref_map[map_idx].uid
            self.curr_frame[obs_idx].matched_obj_score = float(score)
            self.curr_frame[obs_idx].matched_obj_idx = map_idx
            assigned_obs.add(obs_idx)
            assigned_map.add(map_idx)
            match_count += 1

        self.last_added_new_objects = len_curr - match_count
        logger.info(
            "[Tracker][Global] Matched %d/%d observations, added %d new objects.",
            match_count,
            len_curr,
            self.last_added_new_objects,
        )

    def find_overlapping_ratio_faiss(self, pcd1, pcd2, radius=0.02):
        """
        Calculate the percentage of overlapping points between two point clouds using FAISS.

        Parameters:
        pcd1 (numpy.ndarray): Point cloud 1, shape (n1, 3).
        pcd2 (numpy.ndarray): Point cloud 2, shape (n2, 3).
        radius (float): Radius for KD-Tree query (adjust based on point density).

        Returns:
        float: Overlapping ratio between 0 and 1.
        """
        if (
            type(pcd1) == o3d.geometry.PointCloud
            and type(pcd2) == o3d.geometry.PointCloud
        ):
            pcd1 = np.asarray(pcd1.points)
            pcd2 = np.asarray(pcd2.points)

        if pcd1.shape[0] == 0 or pcd2.shape[0] == 0:
            return 0

        # Create the FAISS index for each point cloud
        index1 = faiss.IndexFlatL2(pcd1.shape[1])
        index2 = faiss.IndexFlatL2(pcd2.shape[1])
        index1.add(pcd1.astype(np.float32))
        index2.add(pcd2.astype(np.float32))

        # Query all points in pcd1 for nearby points in pcd2
        D1, I1 = index2.search(pcd1.astype(np.float32), k=1)
        D2, I2 = index1.search(pcd2.astype(np.float32), k=1)

        number_of_points_overlapping1 = np.sum(D1 < radius**2)
        number_of_points_overlapping2 = np.sum(D2 < radius**2)

        overlapping_ratio = np.max(
            [
                number_of_points_overlapping1 / pcd1.shape[0],
                number_of_points_overlapping2 / pcd2.shape[0],
            ]
        )

        return overlapping_ratio

    def compute_box_volume_torch(self, box):
        # box shape is (M, 8, 3)
        edge1 = torch.norm(box[:, 1] - box[:, 0], dim=-1)
        edge2 = torch.norm(box[:, 3] - box[:, 0], dim=-1)
        edge3 = torch.norm(box[:, 4] - box[:, 0], dim=-1)
        return edge1 * edge2 * edge3

    def compute_intersection_volume_torch(self, bbox1, bbox2):
        # Calculate min/max corners for intersection computation
        min_corner1 = torch.min(bbox1, dim=1).values  # Shape (M, 3)
        max_corner1 = torch.max(bbox1, dim=1).values  # Shape (M, 3)
        min_corner2 = torch.min(bbox2, dim=1).values  # Shape (N, 3)
        max_corner2 = torch.max(bbox2, dim=1).values  # Shape (N, 3)

        # Broadcasting for pairwise intersection
        min_intersection = torch.maximum(
            min_corner1[:, None], min_corner2
        )  # Shape (M, N, 3)
        max_intersection = torch.minimum(
            max_corner1[:, None], max_corner2
        )  # Shape (M, N, 3)
        intersection_dims = torch.clamp(
            max_intersection - min_intersection, min=0
        )  # Shape (M, N, 3)
        return torch.prod(intersection_dims, dim=-1)  # Shape (M, N)

    def compute_3d_iou_batch(self, bbox1, bbox2):
        """
        Optimized IoU computation between two sets of axis-aligned 3D bounding boxes using PyTorch.

        bbox1: (M, 8, 3) tensor
        bbox2: (N, 8, 3) tensor

        returns: (M, N) tensor of IoU values
        """
        # Ensure inputs are torch tensors
        if not torch.is_tensor(bbox1):
            bbox1 = torch.tensor(bbox1, dtype=torch.float32)
        if not torch.is_tensor(bbox2):
            bbox2 = torch.tensor(bbox2, dtype=torch.float32)

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bbox1 = bbox1.to(device)
        bbox2 = bbox2.to(device)

        # Compute volumes
        volume1 = self.compute_box_volume_torch(bbox1)  # Shape (M,)
        volume2 = self.compute_box_volume_torch(bbox2)  # Shape (N,)

        # Compute intersection volumes
        intersection_volume = self.compute_intersection_volume_torch(
            bbox1, bbox2
        )  # Shape (M, N)

        # Compute IoU
        iou = intersection_volume / (volume1[:, None] + volume2 - intersection_volume)
        return iou.cpu()
