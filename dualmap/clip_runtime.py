from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOBILECLIP_SUBMODULE = PROJECT_ROOT / "3rdparty" / "mobileclip"


def is_mobileclip_model(model_name: str) -> bool:
    return "MobileCLIP" in model_name


def get_open_clip_model_kwargs(model_name: str) -> dict:
    if model_name.startswith("MobileCLIP2") and not (
        model_name.endswith("S3")
        or model_name.endswith("S4")
        or model_name.endswith("L-14")
    ):
        return {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}
    return {}


def ensure_mobileclip_runtime(model_name: str | None = None) -> None:
    if model_name is not None and not is_mobileclip_model(model_name):
        return

    submodule_path = str(MOBILECLIP_SUBMODULE)
    if MOBILECLIP_SUBMODULE.exists() and submodule_path not in sys.path:
        sys.path.insert(0, submodule_path)


def reparameterize_model(model: torch.nn.Module) -> torch.nn.Module:
    # Local copy of the MobileCLIP reparameterization helper to avoid a separate
    # editable install for the git submodule.
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model


def create_open_clip_components(
    model_name: str,
    pretrained: str,
    device: str,
) -> tuple[torch.nn.Module, object, object]:
    import open_clip

    ensure_mobileclip_runtime(model_name)

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        **get_open_clip_model_kwargs(model_name),
    )
    clip_model = clip_model.to(device)
    clip_model.eval()

    if is_mobileclip_model(model_name):
        clip_model = reparameterize_model(clip_model)

    clip_tokenizer = open_clip.get_tokenizer(model_name)
    return clip_model, clip_preprocess, clip_tokenizer
