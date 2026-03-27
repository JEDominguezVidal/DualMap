#!/usr/bin/env bash
set -euo pipefail

set +u
source /opt/ros/jazzy/setup.bash
set -u

export HF_HOME="${HF_HOME:-/opt/dualmap/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/opt/dualmap/cache/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/opt/dualmap/cache/matplotlib}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/opt/dualmap/cache/ultralytics}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/opt/dualmap/cache/xdg}"
export PYTHONPATH="/opt/dualmap:/opt/dualmap/3rdparty/mobileclip${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}" "${YOLO_CONFIG_DIR}" "${XDG_CACHE_HOME}"

bundled_hf_home="/opt/dualmap/bundled-cache/huggingface"
if [[ -d "${bundled_hf_home}" ]] && \
   [[ -n "$(find "${bundled_hf_home}" -mindepth 1 -print -quit 2>/dev/null)" ]] && \
   [[ -z "$(find "${HF_HOME}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "[entrypoint] Seeding Hugging Face cache from bundled image assets..."
    cp -a "${bundled_hf_home}/." "${HF_HOME}/"
fi

required_paths=(
    "/opt/dualmap/3rdparty/mobileclip/mobileclip/__init__.py"
    "/opt/dualmap/model/yolov8l-world.pt"
    "/opt/dualmap/model/mobile_sam.pt"
    "/opt/dualmap/model/FastSAM-s.pt"
)

for path in "${required_paths[@]}"; do
    if [[ ! -e "${path}" ]]; then
        echo "[entrypoint] Missing required runtime asset: ${path}" >&2
        exit 1
    fi
done

cd /opt/dualmap
exec "$@"
