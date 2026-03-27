#!/usr/bin/env python3

import os

import open_clip


def get_model_kwargs(model_name: str) -> dict:
    if model_name.startswith("MobileCLIP2") and not (
        model_name.endswith("S3")
        or model_name.endswith("S4")
        or model_name.endswith("L-14")
    ):
        return {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}
    return {}


def main() -> None:
    model_name = os.environ.get("CLIP_MODEL_NAME", "MobileCLIP2-S2")
    pretrained = os.environ.get("CLIP_PRETRAINED", "dfndr2b")

    print(
        f"[docker] Prefetching OpenCLIP weights for model={model_name} pretrained={pretrained}"
    )
    open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        **get_model_kwargs(model_name),
    )
    open_clip.get_tokenizer(model_name)
    print("[docker] OpenCLIP prefetch complete.")


if __name__ == "__main__":
    main()

