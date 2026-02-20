import copy
import json
import os
import sys

# Add the project's root directory to sys.path to enable module imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import hydra
import matplotlib
import numpy as np
import open3d as o3d
import open_clip
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

# Intercept and remove custom argument to avoid Hydra conflicts
global_top_k = 5
if "--top_k" in sys.argv:
    idx = sys.argv.index("--top_k")
    if idx + 1 < len(sys.argv):
        global_top_k = int(sys.argv[idx + 1])
        sys.argv.pop(idx)  # remove --top_k flag
        sys.argv.pop(idx)  # remove its value

from utils.object import BaseObject


@hydra.main(version_base=None, config_path="../config/", config_name="query_config")
def main(cfg: DictConfig):

    ### Loading Color map: class id --> color Dict
    if cfg.yolo.use_given_classes:
        given_classes_path = cfg.yolo.given_classes_path
        dir_path = os.path.dirname(given_classes_path)  # './model'
        base_name = os.path.basename(given_classes_path)  # 'gpt_indoor_table.txt'
        file_root, _ = os.path.splitext(base_name)  # 'gpt_indoor_table'

        class_id_colors_path = os.path.join(dir_path, file_root + "_id_colors.json")

    else:
        class_id_colors_path = os.path.join(
            cfg.output_path,
            f"{cfg.dataset_name}_{cfg.scene_id}",
            "classes_info",
            f"{cfg.dataset_name}_{cfg.scene_id}_id_colors.json",
        )

    print("Loading classes id --> colors from: {}".format(class_id_colors_path))

    if not os.path.exists(class_id_colors_path):
        raise FileNotFoundError(f"Error: File not found: {class_id_colors_path}")

    class_id_colors = {}
    with open(class_id_colors_path, "r") as file:
        class_id_colors = json.load(file)
    class_id_colors = {int(key): value for key, value in class_id_colors.items()}

    # Dict: class id --> name
    class_id_names = {}

    if cfg.yolo.use_given_classes:
        class_id_names_path = cfg.yolo.given_classes_path
        # Load class names from txt
        with open(class_id_names_path, "r") as f:
            class_list = [line.strip() for line in f if line.strip()]
        class_id_names = {i: name for i, name in enumerate(class_list)}
    else:
        class_id_names_path = os.path.join(
            cfg.output_path,
            f"{cfg.dataset_name}_{cfg.scene_id}",
            "classes_info",
            f"{cfg.dataset_name}_{cfg.scene_id}_id_names.json",
        )

        with open(class_id_names_path, "r") as file:
            class_id_names = json.load(file)
        class_id_names = {int(key): value for key, value in class_id_names.items()}

    print("Loading classes id --> names  from: {}".format(class_id_names_path))

    if not os.path.exists(class_id_names_path):
        raise FileNotFoundError(f"Error: File not found: {class_id_names_path}")

    ### Loading saved results
    # if map_dir is not provided, use the default path
    load_dir = None
    if os.path.exists(cfg.map_dir):
        load_dir = cfg.map_dir
    else:
        load_dir = os.path.join(
            cfg.output_path, f"{cfg.dataset_name}_{cfg.scene_id}", "map"
        )

    if not os.path.exists(load_dir):
        print(f"Error: {load_dir} does not exist.")
        sys.exit(1)

    print(("Loading saved obj results from: {}".format(load_dir)))

    ### Loading viewpoint
    viewpoint_path = os.path.join(load_dir, "viewpoint.json")
    print(f"Loading viewpoint from: {viewpoint_path}")

    # traverse the .pkl in the directory to get constructed maps
    obj_map = []
    for file in os.listdir(load_dir):
        if file.endswith(".pkl"):
            obj_results_path = os.path.join(load_dir, file)
            # object construction
            loaded_obj = BaseObject.load_from_disk(obj_results_path)
            obj_map.append(loaded_obj)
    print(f"Successfully loaded {len(obj_map)} objects")

    ### Init of CLIP
    print("Loading CLIP model")
    # clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    #     "ViT-H-14", "laion2b_s32b_b79k"
    # )
    # clip_model = clip_model.to("cuda")
    # clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    # MobileCLIP2 S0/S2/B models need custom image normalization
    model_kwargs = {}
    model_name = cfg.clip.model_name
    if model_name.startswith("MobileCLIP2") and not (
        model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14")
    ):
        model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        cfg.clip.model_name, pretrained=cfg.clip.pretrained, **model_kwargs
    )
    clip_model = clip_model.to(cfg.device)
    clip_model.eval()

    # Only reparameterize if the model is MobileCLIP
    if "MobileCLIP" in cfg.clip.model_name:
        from mobileclip.modules.common.mobileone import reparameterize_model

        clip_model = reparameterize_model(clip_model)

    clip_tokenizer = open_clip.get_tokenizer(cfg.clip.model_name)

    print("Done initializing CLIP model.")

    cmap = matplotlib.colormaps.get_cmap("turbo")

    ### Set the visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    # Create window
    vis.create_window(window_name=f"Offline Visualization", width=1920, height=1920)

    for obj in obj_map:
        vis.add_geometry(obj.pcd)

    print(f"Obj Map length: %d" % len(obj_map))

    # Save original RGB colours for rapid switching
    for obj in obj_map:
        obj.original_colors = np.asarray(obj.pcd.colors).copy()

    view_param = None
    if os.path.exists(viewpoint_path):
        print(f"Loading saved viewpoint from {viewpoint_path}")
        view_param = o3d.io.read_pinhole_camera_parameters(viewpoint_path)
        vis.get_view_control().convert_from_pinhole_camera_parameters(view_param)

    # State class for rapid query colour updates
    class QueryState:
        top_1_idx = None
        similarity_colors = None

    state = QueryState()

    def pcd_sem_color_callback(vis):
        print("Show the Pointcloud with semantic colours")
        for obj in obj_map:
            color = class_id_colors[obj.class_id]
            obj.pcd.paint_uniform_color(color)
            vis.update_geometry(obj.pcd)

    def pcd_rgb_color_callback(vis):
        print("Show the Pointcloud with RGB colours")
        for obj in obj_map:
            obj.pcd.colors = o3d.utility.Vector3dVector(obj.original_colors)
            vis.update_geometry(obj.pcd)

    ### Visualization exit
    def vis_exit_callback(vis):
        print("Exiting visualizer...")
        vis.destroy_window()

    def query_callback(vis):
        text_query = input("Enter your query: ")

        # exit the querying
        if text_query == "exit" or text_query == "quit":
            vis.destroy_window()
            sys.exit(0)

        text_queries = [text_query]

        text_queries_tokenized = clip_tokenizer(text_queries).to("cuda")
        text_query_ft = clip_model.encode_text(text_queries_tokenized)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()

        ## Get stacked clip feats from the map
        values = []
        for obj in obj_map:
            values.append(torch.from_numpy(obj.clip_ft))
        map_clip_fts = torch.stack(values, dim=0).to("cuda")

        ## claculate the cos sim between text clip and map clips
        cos_sim = F.cosine_similarity(text_query_ft.unsqueeze(0), map_clip_fts, dim=-1)

        ## Get top k candidates
        top_k = global_top_k
        top_k = min(top_k, len(obj_map)) # Ensure we don't ask for more than we have
        
        top_k_cos_sim, top_k_idx = torch.topk(cos_sim, top_k, dim=0)
        print(f"Top {top_k} similar objects:")
        for i, (cos_val, idx) in enumerate(
            zip(top_k_cos_sim.tolist(), top_k_idx.tolist())
        ):
            print(
                f"{i+1}. No. {idx} {class_id_names[obj_map[idx].class_id]}: {cos_val:.3f}"
            )

        ## Save explicitly the #1 match for highlighting
        state.top_1_idx = top_k_idx.tolist()[0]
        
        max_value = cos_sim.max()
        min_value = cos_sim.min()
        normalized_similarities = (cos_sim - min_value) / (max_value - min_value)
        state.similarity_colors = cmap(normalized_similarities.detach().cpu().numpy())[
            ..., :3
        ]
        
        highlight_objs_callback(vis)

    def highlight_objs_callback(vis):
        print("Highlighting top match in red")
        for idx, obj in enumerate(obj_map):
            if idx == state.top_1_idx:
                obj.pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red
            else:
                obj.pcd.colors = o3d.utility.Vector3dVector(obj.original_colors)
            vis.update_geometry(obj.pcd)

    def queried_color_objs_callback(vis):
        print("Switching to similarity heat map")
        if state.similarity_colors is None:
            return
        for idx, obj in enumerate(obj_map):
            color = state.similarity_colors[idx]
            obj.pcd.paint_uniform_color(color.tolist())
            vis.update_geometry(obj.pcd)

    def help_callback(vis):
        help_info = """
        Keybindings:
        Q - Quit the application
        R - Display the point cloud with RGB colors
        C - Display the point cloud with semantic colors
        F - Enter a query to find top similarity objects
        H - Display this help message
        N - Highlight objects based on previous query results
        M - Colored objects based on previous query results
        S - Save the current viewpoint

        Press the corresponding key to perform the action.
        """
        print(help_info)

    def save_view_callback(vis):
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(viewpoint_path, param)
        print(f"Viewpoint saved to {viewpoint_path}")

    def reset_view():
        if view_param is not None:
            vis.get_view_control().convert_from_pinhole_camera_parameters(view_param)

    vis.register_key_callback(ord("Q"), vis_exit_callback)
    vis.register_key_callback(ord("q"), vis_exit_callback)
    vis.register_key_callback(ord("R"), pcd_rgb_color_callback)
    vis.register_key_callback(ord("r"), pcd_rgb_color_callback)
    vis.register_key_callback(ord("C"), pcd_sem_color_callback)
    vis.register_key_callback(ord("c"), pcd_sem_color_callback)
    vis.register_key_callback(ord("F"), query_callback)
    vis.register_key_callback(ord("f"), query_callback)
    vis.register_key_callback(ord("N"), highlight_objs_callback)
    vis.register_key_callback(ord("n"), highlight_objs_callback)
    vis.register_key_callback(ord("M"), queried_color_objs_callback)
    vis.register_key_callback(ord("m"), queried_color_objs_callback)
    vis.register_key_callback(ord("H"), help_callback)
    vis.register_key_callback(ord("h"), help_callback)
    vis.register_key_callback(ord("S"), save_view_callback)
    vis.register_key_callback(ord("s"), save_view_callback)

    help_callback(vis)

    vis.run()


if __name__ == "__main__":
    main()
