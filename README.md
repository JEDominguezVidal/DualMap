# DualMap
<h3>
  <a href="https://eku127.github.io/DualMap/">Project Page</a> |
  <a href="https://arxiv.org/abs/2506.01950">arXiv</a> 
</h3>

<p align="center">
  <img src="resources/image/optimized-gif.gif" width="70%">
</p>


**DualMap** is an online open-vocabulary mapping system that enables robots to understand and navigate dynamic 3D environments using natural language.

The system supports multiple input sources, including offline datasets (**Dataset Mode**), ROS streams & rosbag files (**ROS Mode**), and iPhone video streams (**Record3d Mode**). We provide examples for each input type.

## News

**[2025.12]**  We have added detailed experiment results of dynamic navigation for better reproduction. Check Table IX and X in the updated [Appendix](https://eku127.github.io/DualMap/static/pdf/appendix_compressed.pdf).

**[2025.08]**  Full code released! 🎉 Welcome to use, share feedback, and contribute.

## Installation

> ✅ Tested on **Ubuntu 24.04** with **ROS 2 Jazzy** and **Python 3.12**

### 1. Clone the Repository (with submodules)

```bash
git clone --branch main --single-branch --recurse-submodules https://github.com/JEDominguezVidal/DualMap.git
cd DualMap
```
> `mobileclip` is no longer installed separately with `pip`, but we still recommend cloning with `--recurse-submodules`.

### 2. Run the Local Setup Script

```bash
./scripts/setup_local_ubuntu24.sh --system-deps
```

This script:

- installs the validated non-ROS Ubuntu packages
- creates `.venv`
- installs all Python dependencies from `requirements.txt`
- runs `python -m scripts.check_install`

If you prefer to install the Ubuntu packages yourself, you can omit `--system-deps`.

> **Note on NumPy**: This project requires `numpy<2.0` (specifically 1.26.x) due to ABI incompatibility with ROS 2 Jazzy's pre-built binaries. The `requirements.txt` file handles this automatically. Do not upgrade numpy to 2.x manually.

### 3. Activate the Environment

```bash
source .venv/bin/activate
```

The separate `pip install -e 3rdparty/mobileclip --no-deps` step is no longer required.

### 4. Runtime Models Download Automatically

The default runtime assets are downloaded automatically on first use if missing:

- `model/yolov8l-world.pt`
- `model/mobile_sam.pt`
- `model/FastSAM-s.pt`

If you want an offline-ready setup up front, run:

```bash
python -m scripts.prefetch_runtime_assets
```

### 5. (Optional) Setup ROS 2 Environment
Setting up ROS2 environment for ROS support and applications.
We recommend [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html).
Once installed, activate the environment:

```bash
source /opt/ros/jazzy/setup.bash
```

> DualMap’s navigation functionality and real-world integration are based on ROS 2. **Installation is strongly recommended**.

> **ROS1 noetic** is also supported, you can setup the ROS 1 in Ubuntu 22.04 by follow [this guide](resources/doc/ros_communication.md).

### 6. (Optional) Setup Habitat Data Collector

[Habitat Data Collector](https://github.com/Eku127/habitat-data-collector) is a tool built on top of the [Habitat-sim](https://github.com/facebookresearch/habitat-sim). It supports agent control, object manipulation, dataset and ROS2 bag recording, as well as navigation through external ROS2 topics. DualMap subscribes to live ROS2 topics from the collector for real-time mapping and language-guided querying, and publishes navigation trajectories for the agent to follow.

> For the best DualMap experience (especially interactive mapping and navigation), **we strongly recommend setting up the Habitat Data Collector**. See [the repo](https://github.com/Eku127/habitat-data-collector) for installation and usage details.

### 7. (Optional) Run with Docker for ROS2

The local Python environment above remains the primary installation path and is fully supported.
In addition, this repository now ships a ROS2-focused Docker workflow that can run the online ROS runner without creating a local virtual environment.

Docker scope in this repository:

- ROS2 Jazzy only
- `python -m applications.runner_ros`
- Both supported pose sources:
  - `nav_msgs/Odometry`
  - `TF` / `TF_STATIC`

The Docker workflow does **not** replace the current `venv` workflow and does **not** cover ROS1 or Record3D in its first version.

See the full guide here:

- [ROS2 Docker Guide](resources/doc/app_runner_ros_docker.md)

Quick start:

```bash
docker build -f docker/Dockerfile.ros2 --target ros2-light -t dualmap:ros2-light .
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose -f docker/compose.ros2.yaml --profile headless up dualmap-ros2-headless
```

The full Docker guide also covers the one-time cleanup of older DualMap Docker images/volumes, the new writable `.docker-cache/` layout, and the GUI default that uses `ros2-bundled`.


## Applications

Here's a quick overview of the requirements for each application type:

| Application | Python Env | ROS1 | ROS2 | Habitat Data Collector |
| :--- | :---: | :---: | :---: | :---: |
| Datasets / Query / iPhone | ✓ | | | |
| ROS (Offline/Online) | ✓ | ✓ | ✓ | |
| Online Sim (Mapping+Nav) | ✓ | | ✓ | ✓ |
* **ROS**: Please install either ROS1 or ROS2 based on your needs.
* **Habitat Data Collector**: Currently, it only supports ROS2.

### 💾 Run with Datasets

DualMap supports running with **offline datasets**. Currently supported datasets include:
1. Replica Dataset  
2. ScanNet Dataset  
3. TUM RGB-D Dataset  
4. Self-collected data using [Habitat Data Collector](https://github.com/Eku127/habitat-data-collector)  

For data collected from your own platform, you can organize it in a similar format to run the system.  

Follow the [Dataset Runner Guide](resources/doc/app_runner_dataset.md) to arrange datasets, run DualMap with these datasets and reproduce our offline mapping results in **Table II** in our paper.

### 🤖 Run with ROS

DualMap supports input from both **ROS1** and **ROS2**. You can run the system with **offline rosbags** or in **online mode** with real robots.

Follow the [ROS Runner Guide](resources/doc/app_runner_ros.md) to get started with running DualMap using ROS1/ROS2 rosbags or live ROS streams.

If you want to run the ROS2 runner through Docker instead of a local Python environment, see the [ROS2 Docker Guide](resources/doc/app_runner_ros_docker.md).

> **Tip**: If you want to use your own rosbags, check out the [Custom Rosbag Guide](resources/doc/app_custom_rosbag.md).

### 🕹️ Online Mapping and Navigation in Simulation

DualMap supports **online** interactive mapping and object navigation in simulation via the [Habitat Data Collector](https://github.com/Eku127/habitat-data-collector).

Follow the [Online Mapping and Navigation Guide](resources/doc/app_simulation.md) to get started with running DualMap in interactive simulation scenes and to reproduce the navigation results (both static and dynamic) in **Table III** in our paper.

### 📱 Run with iPhone

DualMap supports **real-time data streaming** from the **Record3D** app on iPhone.

Follow the [iPhone Runner Guide](resources/doc/app_runner_record_3d.md) to get started with setting up Record3D, streaming data to DualMap, and mapping with your own iPhone!

### 🔍 Offline Map Query

The semantic maps generated by DualMap (whether from real-time ROS streams, custom rosbags, or datasets) can be visually inspected and searched using our Open3D-based query tool. We also provide two prebuilt map examples for testing.

Follow the [Offline Query Guide](resources/doc/app_offline_query.md) to run the query application.

### 🖼️ Visualization
<p align="center">
    <img src="resources/image/app_visual.jpg" width="100%">
</p>

The system supports both [Rerun](https://rerun.io) and [Rviz](http://wiki.ros.org/rviz) visualization. When running with ROS, you can switch the visualizaiton via `use_rerun` and `use_rviz` option in `config/runner_ros.yaml`


## Citation

If you find our work helpful, please consider starring this repo 🌟 and cite:

```bibtex
@ARTICLE{jiang2025dualmap,
  author={Jiang, Jiajun and Zhu, Yiming and Wu, Zirui and Song, Jie},
  journal={IEEE Robotics and Automation Letters},
  title={DualMap: Online Open-Vocabulary Semantic Mapping for Natural Language Navigation in Dynamic Changing Scenes},
  year={2025},
  volume={10},
  number={12},
  pages={12612--12619},
  doi={10.1109/LRA.2025.3621942}
}
```

## Contact
For technical questions, please create an issue. For other questions, please contact the first author: jjiang127 [at] connect.hkust-gz.edu.cn

## Acknowledgment

We are grateful to the authors of [HOVSG](https://github.com/hovsg/HOV-SG) and [ConceptGraphs](https://github.com/concept-graphs/concept-graphs) for their contributions and inspiration.

Special thanks to @[TOM-Huang](https://github.com/Tom-Huang) for his valuable advice and support throughout the development of this project.

We also thank the developers of [MobileCLIP](https://github.com/apple/ml-mobileclip), [CLIP](https://github.com/openai/CLIP), [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM), and [YOLO-World](https://github.com/AILab-CVC/YOLO-World) for their excellent open-source work, which provided strong technical foundations for this project.
