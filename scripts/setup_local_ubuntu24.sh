#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
INSTALL_SYSTEM_DEPS=0

for arg in "$@"; do
    case "$arg" in
        --system-deps)
            INSTALL_SYSTEM_DEPS=1
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Usage: $0 [--system-deps]" >&2
            exit 1
            ;;
    esac
done

if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
        echo "[warn] This setup script is validated for Ubuntu 24.04; detected ${PRETTY_NAME:-unknown}."
    fi
fi

if (( INSTALL_SYSTEM_DEPS )); then
    sudo apt update
    sudo apt install -y \
        git \
        python3.12-venv \
        python3-pip \
        python3.12-dev \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxcursor1 \
        libxi6 \
        libxinerama1
fi

if ! command -v python3.12 >/dev/null 2>&1; then
    echo "[fail] python3.12 is required but was not found in PATH." >&2
    exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    python3.12 -m venv "${VENV_DIR}"
fi

cd "${ROOT_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${ROOT_DIR}/requirements.txt"
python -m scripts.check_install

cat <<'EOF'

Next steps:
  1. source .venv/bin/activate
  2. source /opt/ros/jazzy/setup.bash    # only if you need ROS2
  3. python -m applications.runner_dataset

If you need an offline-ready install, run:
  python -m scripts.prefetch_runtime_assets
EOF
