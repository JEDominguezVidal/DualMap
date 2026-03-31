from __future__ import annotations

import argparse

from dualmap.runtime_assets import ensure_runtime_assets, prefetch_openclip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefetch DualMap runtime model assets for offline-ready use."
    )
    parser.add_argument("--yolo-path", default="model/yolov8l-world.pt")
    parser.add_argument("--sam-path", default="model/mobile_sam.pt")
    parser.add_argument("--fastsam-path", default="model/FastSAM-s.pt")
    parser.add_argument(
        "--skip-openclip",
        action="store_true",
        help="Skip prefetching the default OpenCLIP weights.",
    )
    parser.add_argument("--clip-model-name", default="MobileCLIP2-S2")
    parser.add_argument("--clip-pretrained", default="dfndr2b")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_runtime_assets([args.yolo_path, args.sam_path, args.fastsam_path])
    if not args.skip_openclip:
        prefetch_openclip(args.clip_model_name, args.clip_pretrained)
    print("[ok] Runtime asset prefetch completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
