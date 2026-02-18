# Using Custom Rosbags with DualMap

This guide explains how to configure DualMap to work with your own ROS2 rosbags.

## 1. Prepare Configuration File

To tell DualMap which topics to use, you must modify the configuration file:  
`config/data_config/ros/ros2_rosbag.yaml`

This file defines the topic names, depth scaling, camera intrinsics, and extrinsics.

### 1.1 Topic Names
Open your rosbag info to check the available topics:
```bash
ros2 bag info path/to/your/rosbag
```

Updated the `ros_topics` section in `ros2_rosbag.yaml`:
```yaml
ros_topics:
  rgb: "/your/rgb/topic"       # e.g., /camera/color/image_raw
  depth: "/your/depth/topic"   # e.g., /camera/depth/image_raw
  odom: "/your/odom/topic"     # e.g., /odom or /robot/odom
```

> [!IMPORTANT]
> **Odometry Requirement**: DualMap requires a `nav_msgs/Odometry` topic for localisation. If your rosbag only records TF (Transformations), you must run a separate ROS node to convert TF to Odometry or publish the odometry topic before running DualMap. The system **does not** subscribe to the `/tf` topic directly.

### 1.2 Depth Factor
You must set the correct `depth_factor` to convert your depth image values to **meters**.
- If your depth image is **16-bit integer (uint16)** and represents **millimetres** (common in RealSense, Orbbec): set `1000.0`.
- If your depth image is **32-bit float** and represents **meters** (common in simulators): set `1.0`.

### 1.3 Camera Intrinsics
The system needs the camera intrinsic parameters ($f_x, f_y, c_x, c_y$).
There are two ways to provide them:

**Option A: Manual Configuration (Recommended)**
Fill in the `intrinsic` section in `ros2_rosbag.yaml`.
```yaml
intrinsic:
  fx: 600.0
  fy: 600.0
  cx: 320.0
  cy: 240.0
```
> [!WARNING]
> The intrinsic parameters **MUST match the resolution** of the input images. For example, if your image is $640 \times 480$, $c_x$ should be around 320 and $c_y$ around 240. If you resize the images, you must scale these parameters accordingly.

**Option B: Camera Info Topic**
If you leave the `intrinsic` section commented out or empty, the system will try to read from the `camera_info` topic specified in the config.

### 1.4 Camera Extrinsics (Base -> Camera)
This defines the static transformation from the **Robot Base Frame** (defined by your Odom topic) to the **Camera Optical Frame**.

- If your camera is mounted exactly at the robot's base frame origin (or if your Odom topic already represents the camera pose), you can leave `extrinsics` as an Identity matrix.
- If your camera is offset (e.g., mounted high up and tilted down), you **must** provide the $4 \times 4$ transformation matrix.

```yaml
# Example: Camera is 0.5m above base and 0.2m forward
extrinsics:
  - [1.0, 0.0, 0.0, 0.2]
  - [0.0, 1.0, 0.0, 0.0]
  - [0.0, 0.0, 1.0, 0.5]
  - [0.0, 0.0, 0.0, 1.0]
```
> [!IMPORTANT]
> DualMap does **not** rely on the ROS TF tree to determine the camera position relative to the base. This static matrix is the only way to tell the system where the camera is.

---

## 2. Configure Runner

Once your data config is ready, tell the system to use it.

Open `config/runner_ros.yaml` and update the `ros_stream_config_path`:

```yaml
# Path to ROS topic configuration file
ros_stream_config_path: ./config/data_config/ros/ros2_rosbag.yaml
```

Also, check `use_compressed_topic`:
- Set to `true` if your topics are `sensor_msgs/CompressedImage`.
- Set to `false` if they are raw `sensor_msgs/Image`.

---

## 3. Verify System Config (Classes)

Ensure the system is looking for the classes/objects relevant to your environment.
Open `config/system_config.yaml`:

```yaml
# Choose a class list appropriate for your scene
given_classes_path: ./config/class_list/gpt_indoor_office.txt
```
You can create your own class list text file if needed.

---

## 4. Run the System

1.  **Start DualMap**:
    ```bash
    cd DualMap
    source dualmap312/bin/activate
    source /opt/ros/jazzy/setup.bash
    python -m applications.runner_ros
    ```

2.  **Play your Rosbag**:
    In a separate terminal:
    ```bash
    source /opt/ros/jazzy/setup.bash
    ros2 bag play path/to/your/rosbag
    ```
