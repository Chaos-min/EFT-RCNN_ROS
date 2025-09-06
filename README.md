# video
look: https://www.bilibili.com/video/BV1g5aqznE8x/?spm_id_from=333.1387.list.card_archive.click&vd_source=ef216662158dd7b08a506691db35dcf0

# EFT-RCNN_ROS
Robust Pedestrian Detection and Intrusion Judgment in Coal Yard Hazard Areas via 3D LiDAR-Based Deep Learning. 
A 3D detection EFT-RCNN ROS deployment on NVIDIA 4060 (8GB)

# Installation
## Requirements
the codes are tested in the following environment: 
1. Linux(test on ubuntu 22.04/20.04);
2. ROS(noetic);
3. Python 3.9, PyTorch 2.1, CUDA 11.8;
4. spconv 2.3.6;

## Install
1. You need build conda env for eft-rcnn, this model based on OpenPCDet, look:https://github.com/open-mmlab/OpenPCDet.
2. You need build ros-noetic, and create a workspace for detection model look:https://github.com/BIT-DYN/pointpillars_ros
3. We provide point cloud data for your test (from QT128, or you can use kitti dataset), look:

# Usage

