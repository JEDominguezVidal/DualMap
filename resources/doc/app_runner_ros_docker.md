# Running DualMap with Docker (ROS2 Jazzy)

This guide adds a Docker workflow for the ROS2 runner without replacing the current local installation with `venv`.

## Scope

The Docker workflow in this repository is focused on:

- ROS2 Jazzy
- `python -m applications.runner_ros`
- Both ROS pose contracts already supported by the codebase:
  - `nav_msgs/Odometry`
  - `TF` / `TF_STATIC`

The following remain outside the first Docker workflow:

- ROS1
- `runner_record_3d`
- Direct camera-driver containerization
- Dataset/query applications

In other words, the container is meant to run DualMap itself and subscribe to an existing ROS2 graph on the host.

## What gets built

The Docker setup provides two image targets in `docker/Dockerfile.ros2`:

- `ros2-light`: lighter image, uses an external cache for OpenCLIP/Hugging Face at runtime
- `ros2-bundled`: prefetches the default CLIP weights (`MobileCLIP2-S2`, `dfndr2b`) during build

Compose defaults:

- GUI profile -> `ros2-bundled`
- Headless profile -> `ros2-light`

Both targets:

- install ROS2 Jazzy user-space dependencies inside the image
- install Python runtime dependencies from a pinned Docker lock file
- expose `mobileclip` from the git submodule through `PYTHONPATH`
- ensure that `YOLO-World`, `MobileSAM`, and `FastSAM` weights exist in `/opt/dualmap/model`

If the local repository already has those `.pt` files, the build reuses them. If not, the build downloads them through Ultralytics.

Persistent runtime caches are stored in repo-local bind mounts under `./.docker-cache/` so they stay writable for the host user running the container.

## Prerequisites

1. Linux host with Docker
2. ROS2 Jazzy topics available on the host
3. For GPU execution:
   - NVIDIA driver installed on the host
   - NVIDIA Container Toolkit configured

Optional but recommended for GUI mode:

- X11 desktop session
- `xhost` available on the host

## One-time cleanup from the previous Docker workflow

If you tested an older revision of the Docker setup that used named volumes, run this cleanup once before rebuilding:

```bash
docker compose -f docker/compose.ros2.yaml down -v --remove-orphans
docker image rm dualmap:ros2-light dualmap:ros2-bundled || true
docker volume rm \
  docker_dualmap_hf_cache \
  docker_dualmap_torch_cache \
  docker_dualmap_matplotlib_cache \
  docker_dualmap_ultralytics_cache \
  docker_dualmap_xdg_cache || true
docker builder prune -af
```

This removes only old DualMap images, old DualMap cache volumes, and unused build cache. It does not touch unrelated images.

## Build

From the repository root:

```bash
docker build \
  -f docker/Dockerfile.ros2 \
  --target ros2-bundled \
  -t dualmap:ros2-bundled \
  .
```

To build the headless image too:

```bash
docker build \
  -f docker/Dockerfile.ros2 \
  --target ros2-light \
  -t dualmap:ros2-light \
  .
```

Compose can also build the correct default target for each profile:

```bash
docker compose -f docker/compose.ros2.yaml --profile gui build dualmap-ros2-gui
docker compose -f docker/compose.ros2.yaml --profile headless build dualmap-ros2-headless
```

Optional overrides:

- `DUALMAP_GUI_TARGET=ros2-light` changes the GUI service target
- `DUALMAP_HEADLESS_TARGET=ros2-bundled` changes the headless service target

## Run Headless

The headless profile disables `Rerun` and `RViz` through Hydra overrides:

```bash
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) \
docker compose -f docker/compose.ros2.yaml --profile headless up dualmap-ros2-headless
```

This service uses:

- `network_mode: host` for ROS2 DDS discovery
- `gpus: all`
- mounted `config`, `output`, and `outputs`
- writable bind-mounted caches in `./.docker-cache/`

## Run with GUI

Allow the container to access your X server:

```bash
xhost +local:
```

Then start the GUI profile:

```bash
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) \
docker compose -f docker/compose.ros2.yaml --profile gui up dualmap-ros2-gui
```

This profile mounts `/tmp/.X11-unix`, forwards `DISPLAY`, and uses the bundled image by default so the CLIP weights are available without a runtime download.

If you prefer to keep the Rerun server inside Docker but open the native viewer manually on the host:

```bash
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) \
docker compose -f docker/compose.ros2.yaml --profile gui run --rm \
  dualmap-ros2-gui \
  python3 -m applications.runner_ros spawn_rerun_viewer=false
```

Then connect from the host:

```bash
rerun --connect rerun+http://127.0.0.1:9876/proxy
```

When you finish testing, you can revert the X11 permission change:

```bash
xhost -local:
```

## Using your own ROS2 configuration

The Compose services mount the local `config/` directory into the container, so your current YAML workflow is preserved.

Examples:

- `config/runner_ros.yaml`
- `config/data_config/ros/realsense.yaml`
- `config/data_config/ros/realsense_tf.yaml`

If you need to override parameters at launch time, pass extra Hydra arguments by overriding the command:

```bash
docker compose -f docker/compose.ros2.yaml run --rm \
  dualmap-ros2-headless \
  python3 -m applications.runner_ros ros_stream_config_path=./config/data_config/ros/realsense_tf.yaml
```

## Notes on outputs and caches

- Final outputs still go to the same relative project paths used today:
  - `output/map_results`
  - `outputs/`
- Runtime caches are stored in:
  - `.docker-cache/huggingface`
  - `.docker-cache/torch`
  - `.docker-cache/matplotlib`
  - `.docker-cache/ultralytics`
  - `.docker-cache/xdg`
- The bundled target seeds the runtime Hugging Face cache automatically on first start.
- If you change `clip.model_name` to a different model, the bundled cache may no longer be sufficient and the container may need to download the new weights.

## Validation commands

Import smoke test:

```bash
docker run --rm --gpus all dualmap:ros2-light \
  python3 -c "import rclpy, cv_bridge, tf2_ros, message_filters, torch, open_clip, open3d, ultralytics"
```

Unit tests:

```bash
docker run --rm --gpus all dualmap:ros2-light \
  python3 -m unittest discover -s tests -p 'test_*.py'
```

## Troubleshooting

### No ROS2 topics discovered

Make sure the publisher is on the host and the container is started with `network_mode: host`.

### GUI does not open

Check:

- `DISPLAY` is set on the host
- `xhost +local:` was executed before starting the GUI profile
- you are using the `gui` profile, not `headless`

If you still prefer not to auto-open the viewer inside Docker, run with `spawn_rerun_viewer=false` and connect from the host with:

```bash
rerun --connect rerun+http://127.0.0.1:9876/proxy
```

### CLIP downloads again even with the bundled image

That usually means one of these happened:

- the runtime cache under `.docker-cache/huggingface` was cleared
- you changed `clip.model_name`
- you changed `clip.pretrained`
