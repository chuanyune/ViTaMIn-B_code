<h1 align="center">
ViTaMIn-B:
A Reliable and Efficient Visuo-Tactile BiManual Manipulation Interface
</h1>

<p align="center">
    <a href="https://chuanyune.github.io/">Chuanyu Li</a><sup>1*</sup>,
    Chaoyi Liu<sup>1*</sup>,
    Daotan Wang<sup>1</sup>,
    Shuyu Zhang<sup>4</sup>,
    <br>
    Lusong Li<sup>3</sup>,
    Zecui Zeng<sup>3</sup>,
    <a href="https://fangchenliu.github.io/">Fangchen Liu</a><sup>2</sup>,
    Jing Xu<sup>1†</sup>,
    <a href="https://callmeray.github.io/homepage/">Rui Chen</a><sup>1†</sup>
    <br>
    <sup>1</sup>Tsinghua University &nbsp;&nbsp;
    <sup>2</sup>University of California, Berkeley
    <br>
    <sup>3</sup>JD Explore Academy &nbsp;&nbsp;
    <sup>4</sup>The Hong Kong Polytechnic University
    <br>
    <sup>*</sup>Equal contribution &nbsp;&nbsp;
    <sup>†</sup>Equal advising
</p>

<div align="center">
<a href='https://chuanyune.github.io/ViTaMIn-B_page/'><img alt='arXiv' src='https://img.shields.io/badge/arXiv-2503.02881-red.svg'></a>     
<a href='https://chuanyune.github.io/ViTaMIn-B_page/'><img alt='project website' src='https://img.shields.io/website-up-down-green-red/http/cv.lbesson.qc.to.svg'></a>     
<a href='https://huggingface.co/datasets/chuanyune/ViTaMIn-B_data_and_ckpt/tree/main'><img alt='data' src='https://img.shields.io/badge/data-FFD21E?logo=huggingface&logoColor=000'></a>  
<a href='https://huggingface.co/datasets/chuanyune/ViTaMIn-B_data_and_ckpt/tree/main'><img alt='checkpoints' src='https://img.shields.io/badge/checkpoints-FFD21E?logo=huggingface&logoColor=000'></a>    
</div>

---

## 🔧 Hardware Setup

We provide multiple hardware configuration options for data collection:

### Visual-Only Data Collection
- **UMI Gripper**: Use the standard gripper from the UMI system (3D-printable gripper models can be downloaded from the [UMI](https://real-stanford.github.io/universal_manipulation_interface/))

### Visual + Tactile Data Collection
We offer three tactile sensor options (grippers require assembly following our provided instructions and molds):
1. **AllTact Gripper**: Vision-based tactile sensor with custom gripper design
2. **DuoTact Gripper**: Dual tactile sensor setup with custom gripper design
3. **GelSight Sensor**: High-resolution tactile sensing solution

For detailed hardware components, assembly instructions, and mold files, please visit our [project page](https://chuanyune.github.io/ViTaMIn-B_page/).

Once you've assembled the data collection device, connect the cameras and foot pedal to your computer.

## 📦 Installation

> **System Requirements:** Ubuntu 20.04 or 22.04

### Clone Repository

```bash
git clone git@github.com:chuanyune/ViTaMIn-B.git
cd ViTaMIn-B
```

### Setup Environment

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate vitaminb
```

### Install Tracking App

Since we use Meta Quest for pose tracking, you need to install the tracking app [VB-quest](assets/VB-quest.apk) on your Quest headset via [SideQuest](https://sidequestvr.com/).

## 🎥 Data Collection

### ⚙️ Configuration

In the [config file](config/task_config.yaml), modify the `name` parameter to give your task a unique identifier. 

> **⚠️ Important:** Do not change the `name` parameter once you've started collecting data, as all subsequent pipeline steps depend on it.

### 📹 Step 1: Start Data Recorder

Launch the data recorder:

```bash
python vitamin_b_data_collection_pipeline/00_data_recorder.py --cfg config/task_config.yaml
```

### 🔌 Step 2: Setup ADB Port Forwarding

1. Put on your Quest headset and launch the installed tracking app
2. In the app, select **Y: TCP Mode**
3. Open a new terminal on your computer and run:

```bash
adb forward tcp:7777 tcp:7777
```

> **Troubleshooting:** If ADB forwarding fails, you may need to disable the firewall:
> ```bash
> sudo ufw disable
> ```

If no error is returned, port forwarding is successful.

### 🎯 Step 3: Start Pose Tracking

In the same terminal, launch the pose tracking script:

```bash
python vitamin_b_data_collection_pipeline/00_get_pose.py --cfg config/task_config.yaml
```

You can now start collecting data using the foot pedal.

### 🔄 Step 4: Process Collected Data

After collecting all episodes, run the pipeline to generate training data:

```bash
python run_data_collection_pipeline.py
```

> **Note:** This script sequentially executes all programs in the `vitamin_b_data_collection_pipeline` directory. You can comment out any steps you don't need.


### 📊 Zarr Data Structure

After running the data collection pipeline with `08_generate_replay_buffer.py`, your data will be stored in Zarr format with the following structure:

#### Bimanual Setup Data Format

**Robot State (Left Hand - robot0):**
```
├── robot0_eef_pos (N, 3) float32                  # End-effector position
├── robot0_eef_rot_axis_angle (N, 3) float32      # End-effector rotation
├── robot0_gripper_width (N, 1) float32           # Gripper width
├── robot0_demo_start_pose (N, 7) float32         # Demo start pose
├── robot0_demo_end_pose (N, 7) float32           # Demo end pose
```

**Robot State (Right Hand - robot1):**
```
├── robot1_eef_pos (N, 3) float32                  # End-effector position
├── robot1_eef_rot_axis_angle (N, 3) float32      # End-effector rotation
├── robot1_gripper_width (N, 1) float32           # Gripper width
├── robot1_demo_start_pose (N, 7) float32         # Demo start pose
├── robot1_demo_end_pose (N, 7) float32           # Demo end pose
```

**Vision & Tactile Data (Left Hand - camera0):**
```
├── camera0_rgb (N, H, W, 3) uint8                # Visual camera
├── camera0_left_tactile (N, H_t, W_t, 3) uint8  # Left tactile sensor image
├── camera0_left_tactile_points (N, P, 3) float32 # Left tactile point cloud
├── camera0_right_tactile (N, H_t, W_t, 3) uint8 # Right tactile sensor image
├── camera0_right_tactile_points (N, P, 3) float32 # Right tactile point cloud
```

**Vision & Tactile Data (Right Hand - camera1):**
```
├── camera1_rgb (N, H, W, 3) uint8                # Visual camera
├── camera1_left_tactile (N, H_t, W_t, 3) uint8  # Left tactile sensor image
├── camera1_left_tactile_points (N, P, 3) float32 # Left tactile point cloud
├── camera1_right_tactile (N, H_t, W_t, 3) uint8 # Right tactile sensor image
└── camera1_right_tactile_points (N, P, 3) float32 # Right tactile point cloud
```

#### Variable Definitions

- **N**: Total number of frames across all episodes
- **H × W**: Visual image resolution (set by `visual_out_res` in config)
- **H_t × W_t**: Tactile image resolution (set by `tactile_out_res` in config)
- **P**: Number of points in tactile point cloud (set by `fps_num_points` in config)

#### Important Notes

- **Tactile data** (`*_tactile` and `*_tactile_points`) are only generated when `use_tactile_img` and/or `use_tactile_pc` are enabled in the config. For vision-only policies, set these to `False`.
- **Camera indices**: `camera0` = left hand, `camera1` = right hand
- **Pose format**: `[x, y, z, rx, ry, rz, rw]` where `(x, y, z)` is position and `(rx, ry, rz, rw)` depends on rotation representation
- **Compression**: Images are compressed using JpegXl with configurable compression level


## 🚀 Training Policy

### 🤗 Hugging Face Setup (Optional)

If you experience issues loading models from Hugging Face, configure a mirror:

```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
```

### Training Commands

#### Single GPU Training

```bash
python train.py --config-name=train_vision_tactile_pc
```

#### Multi-GPU Training

For training with 8 GPUs:

```bash
accelerate launch --num_processes 8 train.py --config-name=train_vision_tactile_pc
```

### 🔧 Troubleshooting

**Issue:** `ImportError: cannot import name 'cached_download' from 'huggingface_hub'`

**Solution:** Locate the `dynamic_modules_utils.py` file and remove the import statement for `cached_download`.


## 🤖 Real-World Deployment

Our reference implementation uses the **Rokae xMate ER3 Pro** robot arm with a **PGI gripper**. 

### Hardware Adaptation

To adapt this system to your own robot hardware, modify the following interfaces:
- [Robot Interface](./real_world/rokae/rokae_interface.py) - Robot arm control
- [Gripper Interface](./real_world/pgi/pgi_interface.py) - Gripper control

### Deploy Trained Policy

Run the deployment script with your trained checkpoint:

```bash
python deploy_scripts/eval_real_bimanual_vb.py -i 'path/to/your/ckpt'
```


## 🙏 Acknowledgement

Our work is built upon [UMI](https://github.com/real-stanford/universal_manipulation_interface)
and [ARCap](https://stanford-tml.github.io/ARCap/).
Thanks for their great work!

## 🔗 Citation

If you find our work useful, please consider citing:

```
--
```