from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile
from pathlib import Path

from dualmap.runtime_assets import SUPPORTED_RUNTIME_ASSETS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model"
EXPECTED_PYTHON = (3, 12)
DEFAULT_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "dualmap-matplotlib"
DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))


CORE_IMPORTS = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("open_clip", "open_clip"),
    ("timm", "timm"),
    ("tyro", "tyro"),
    ("wandb", "wandb"),
    ("h5py", "h5py"),
    ("hydra", "hydra"),
    ("omegaconf", "omegaconf"),
    ("distinctipy", "distinctipy"),
    ("ultralytics", "ultralytics"),
    ("dill", "dill"),
    ("supervision", "supervision"),
    ("open3d", "open3d"),
    ("imageio", "imageio"),
    ("natsort", "natsort"),
    ("kornia", "kornia"),
    ("rerun", "rerun"),
    ("record3d", "record3d"),
    ("pyliblzfse", "pyliblzfse"),
    ("png", "png"),
    ("tabulate", "tabulate"),
    ("pympler", "pympler"),
    ("plyfile", "plyfile"),
    ("numpy", "numpy"),
    ("psutil", "psutil"),
    ("faiss", "faiss"),
    ("scipy", "scipy"),
    ("sklearn", "sklearn"),
    ("cv2", "cv2"),
    ("matplotlib", "matplotlib"),
    ("networkx", "networkx"),
    ("yaml", "yaml"),
    ("tqdm", "tqdm"),
    ("pandas", "pandas"),
    ("PIL", "PIL"),
    ("xlsxwriter", "xlsxwriter"),
]


ROS2_IMPORTS = [
    ("rclpy", "rclpy"),
    ("cv_bridge", "cv_bridge"),
    ("message_filters", "message_filters"),
    ("tf2_ros", "tf2_ros"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the local DualMap install.")
    parser.add_argument(
        "--ros2",
        action="store_true",
        help="Also validate ROS2 Jazzy imports after sourcing /opt/ros/jazzy/setup.bash.",
    )
    return parser.parse_args()


def check_python() -> list[str]:
    errors = []
    if sys.version_info[:2] != EXPECTED_PYTHON:
        errors.append(
            "Expected Python "
            f"{EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}, got "
            f"{sys.version_info.major}.{sys.version_info.minor}."
        )
    return errors


def check_imports(imports: list[tuple[str, str]]) -> list[str]:
    errors = []
    for display_name, module_name in imports:
        try:
            importlib.import_module(module_name)
            print(f"[ok] import {display_name}")
        except Exception as exc:  # pragma: no cover - diagnostic entrypoint
            errors.append(f"Failed to import {display_name}: {exc}")
    return errors


def report_assets() -> None:
    print("[info] Default runtime assets:")
    for asset_name in sorted(SUPPORTED_RUNTIME_ASSETS):
        path = DEFAULT_MODEL_DIR / asset_name
        if path.exists() and path.stat().st_size > 0:
            print(f"[ok] {path}")
        else:
            print(
                f"[info] Missing {path} (will auto-download on first run or via "
                "'python -m scripts.prefetch_runtime_assets')."
            )


def check_mobileclip_submodule() -> None:
    mobileclip_dir = PROJECT_ROOT / "3rdparty" / "mobileclip"
    if mobileclip_dir.exists():
        print(f"[ok] Found MobileCLIP submodule at {mobileclip_dir}")
    else:
        print(
            "[warn] 3rdparty/mobileclip is missing. The separate editable install is no "
            "longer required, but keeping the submodule cloned is still recommended for "
            "MobileCLIP runtime compatibility."
        )


def check_ros2() -> list[str]:
    errors = []
    if Path("/opt/ros/jazzy/setup.bash").exists():
        print("[ok] Found /opt/ros/jazzy/setup.bash")
    else:
        errors.append("Missing /opt/ros/jazzy/setup.bash.")

    if "ROS_DISTRO" not in os.environ:
        errors.append(
            "ROS2 environment not sourced. Run 'source /opt/ros/jazzy/setup.bash' first."
        )
        return errors

    if os.environ.get("ROS_DISTRO") != "jazzy":
        errors.append(
            f"Expected ROS_DISTRO=jazzy, got {os.environ.get('ROS_DISTRO')!r}."
        )
    return errors + check_imports(ROS2_IMPORTS)


def main() -> int:
    args = parse_args()

    errors = []
    errors.extend(check_python())
    errors.extend(check_imports(CORE_IMPORTS))
    check_mobileclip_submodule()
    report_assets()

    if args.ros2:
        errors.extend(check_ros2())

    if errors:
        print("[fail] Installation check failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("[ok] DualMap installation looks consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
