from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

SUPPORTED_RUNTIME_ASSETS = {
    "yolov8l-world.pt": "YOLO",
    "mobile_sam.pt": "SAM",
    "sam_l.pt": "SAM",
    "FastSAM-s.pt": "FastSAM",
}


def _asset_factories() -> dict[str, object]:
    from ultralytics import FastSAM, SAM, YOLO

    return {
        "yolov8l-world.pt": YOLO,
        "mobile_sam.pt": SAM,
        "sam_l.pt": SAM,
        "FastSAM-s.pt": FastSAM,
    }


def locate_downloaded_file(temp_dir: Path, asset_name: str, model: object) -> Path:
    candidates = []

    ckpt_path = getattr(model, "ckpt_path", None)
    if isinstance(ckpt_path, str):
        candidates.append(Path(ckpt_path))

    candidates.append(temp_dir / asset_name)
    candidates.extend(temp_dir.rglob(asset_name))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not locate downloaded runtime asset '{asset_name}'."
    )


def ensure_runtime_asset(model_path: str | Path) -> Path:
    dest = Path(model_path)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    asset_name = dest.name
    factories = _asset_factories()
    if asset_name not in factories:
        supported = ", ".join(sorted(SUPPORTED_RUNTIME_ASSETS))
        raise FileNotFoundError(
            f"Missing model file '{dest}'. Automatic download is only supported for: "
            f"{supported}"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="dualmap-ultralytics-"))
    cwd = Path.cwd()

    try:
        os.chdir(temp_dir)
        model = factories[asset_name](asset_name)
        source = locate_downloaded_file(temp_dir, asset_name, model)
        shutil.copy2(source, dest)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to auto-download runtime asset '{asset_name}' to '{dest}'."
        ) from exc
    finally:
        os.chdir(cwd)
        shutil.rmtree(temp_dir, ignore_errors=True)

    return dest


def ensure_runtime_assets(model_paths: list[str | Path]) -> list[Path]:
    return [ensure_runtime_asset(model_path) for model_path in model_paths]


def prefetch_openclip(
    model_name: str = "MobileCLIP2-S2",
    pretrained: str = "dfndr2b",
) -> None:
    import open_clip
    from dualmap.clip_runtime import get_open_clip_model_kwargs

    open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        **get_open_clip_model_kwargs(model_name),
    )
    open_clip.get_tokenizer(model_name)
