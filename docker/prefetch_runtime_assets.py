#!/usr/bin/env python3

import os
import shutil
import tempfile
from pathlib import Path

from ultralytics import FastSAM, SAM, YOLO


ROOT = Path("/opt/dualmap")
MODEL_DIR = ROOT / "model"


def locate_downloaded_file(temp_dir: Path, asset_name: str, model) -> Path:
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


def ensure_asset(asset_name: str, factory) -> None:
    dest = MODEL_DIR / asset_name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[docker] Using existing runtime asset: {dest}")
        return

    print(f"[docker] Downloading missing runtime asset: {asset_name}")
    temp_dir = Path(tempfile.mkdtemp(prefix="dualmap-ultralytics-"))
    cwd = Path.cwd()

    try:
        os.chdir(temp_dir)
        model = factory(asset_name)
        source = locate_downloaded_file(temp_dir, asset_name, model)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        print(f"[docker] Saved runtime asset to: {dest}")
    finally:
        os.chdir(cwd)
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    ensure_asset("yolov8l-world.pt", YOLO)
    ensure_asset("mobile_sam.pt", SAM)
    ensure_asset("FastSAM-s.pt", FastSAM)


if __name__ == "__main__":
    main()

