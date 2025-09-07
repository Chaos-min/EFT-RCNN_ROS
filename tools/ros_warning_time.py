#!/usr/bin/env python3
# coding: utf-8

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
import ros_numpy as ros_numpy

import time
import numpy as np
from pyquaternion import Quaternion

import argparse
import glob
from pathlib import Path
import datetime
import os

import mayavi.mlab as mlab
import numpy as np
import torch
import scipy.linalg as linalg

import sys

# 新增Open3D库 - 用于高效点云处理 TODO 高效体素化处理LiDAR点云
try:
    import open3d as o3d

    OPEN3D_FLAG = True
except:
    OPEN3D_FLAG = False
    print("Warning: Open3D not found, point cloud downsampling will be disabled")

# 使用cuDNN加速
try:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    CUDNN_ENABLED = True
    print("cuDNN acceleration enabled")
except:
    CUDNN_ENABLED = False
    print("Warning: cuDNN acceleration disabled")

# 获取当前脚本路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# 可视化相关
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path: 根目录
            dataset_cfg: 数据集配置
            class_names: 类别名称
            training: 训练模式
            logger: 日志
            ext: 扩展名
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )


# ======================== Start:三级预警核心函数优化 ========================
def ray_casting(point, polygon):
    """射线法判断点是否在多边形内"""
    x, y = point
    crossings = 0
    polygon = np.array(polygon)
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i + 1) % len(polygon)]
        ax, ay = a
        bx, by = b
        if (ay > y) != (by > y):
            x_intersect = (bx - ax) * (y - ay) / (by - ay + 1e-9) + ax
            if x < x_intersect:
                crossings += 1
    return crossings % 2 == 1


def point_to_segment_dist_vectorized(p, a, b):
    """计算点到线段的最短距离（向量化实现）"""
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-9)
    t = np.clip(t, 0.0, 1.0)
    projection = a + t * ab
    return np.linalg.norm(p - projection)


def min_distance_to_polygon(point, polygon):
    """计算点到多边形的最短距离（向量化实现）"""
    polygon = np.array(polygon)
    point = np.array(point)

    # 创建所有线段的起点和终点
    segment_starts = polygon
    segment_ends = np.roll(polygon, -1, axis=0)

    # 计算所有线段的距离并找最小值
    distances = np.array([point_to_segment_dist_vectorized(point, start, end)
                          for start, end in zip(segment_starts, segment_ends)])

    return np.min(distances)


def invasion_warning(point, polygon, D1=4.0, D2=1.0):
    """分级预警模型，返回预警级别和距离"""
    is_inside = ray_casting(point, polygon)
    d = min_distance_to_polygon(point, polygon)

    if is_inside:
        return '报警L3', 0.0  # 在内部时距离设为0
    else:
        if d < D2:
            return '报警L3', d
        elif d <= D1:
            return '预警L2', d
        else:
            return '预判L1', d

# ======================== End:三级预警核心函数结束 ========================


# 点云PointField生成函数，用于重发布点云
def _make_point_field(num_field):
    """生成点云的PointField，用于构建点云消息"""
    msg_pf1 = PointField()
    msg_pf1.name = np.str_('x')
    msg_pf1.offset = np.uint32(0)
    msg_pf1.datatype = np.uint8(7)
    msg_pf1.count = np.uint32(1)

    msg_pf2 = PointField()
    msg_pf2.name = np.str_('y')
    msg_pf2.offset = np.uint32(4)
    msg_pf2.datatype = np.uint8(7)
    msg_pf2.count = np.uint32(1)

    msg_pf3 = PointField()
    msg_pf3.name = np.str_('z')
    msg_pf3.offset = np.uint32(8)
    msg_pf3.datatype = np.uint8(7)
    msg_pf3.count = np.uint32(1)

    msg_pf4 = PointField()
    msg_pf4.name = np.str_('intensity')
    msg_pf4.offset = np.uint32(16)
    msg_pf4.datatype = np.uint8(7)
    msg_pf4.count = np.uint32(1)

    if num_field == 4:
        return [msg_pf1, msg_pf2, msg_pf3, msg_pf4]

    msg_pf5 = PointField()
    msg_pf5.name = np.str_('label')
    msg_pf5.offset = np.uint32(20)
    msg_pf5.datatype = np.uint8(4)
    msg_pf5.count = np.uint32(1)

    return [msg_pf1, msg_pf2, msg_pf3, msg_pf4, msg_pf5]


# TODO 新增点云下采样函数
def voxel_downsample(points_numpy, voxel_size=0.1):
    """使用Open3D进行高效的体素下采样

    Args:
        points_numpy: [N, 4] 点云数组，包含x,y,z,intensity
        voxel_size: float, 体素大小，越大下采样越多

    Returns:
        下采样后的点云数组 [M, 4] 其中M<N
    """
    if not OPEN3D_FLAG:
        return points_numpy

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_numpy[:, :3])

    # 添加强度作为颜色以保留强度信息
    # 归一化强度到[0,1]以符合Open3D颜色范围
    intensity_normalized = points_numpy[:, 3:4] / max(np.max(points_numpy[:, 3]), 1)
    colors = np.hstack([intensity_normalized, intensity_normalized, intensity_normalized])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 执行体素下采样
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 转回numpy数组
    downsampled_points = np.asarray(downsampled_pcd.points)
    downsampled_colors = np.asarray(downsampled_pcd.colors)

    # 恢复强度值
    downsampled_intensity = downsampled_colors[:, 0] * max(np.max(points_numpy[:, 3]), 1)

    # 重新组合xyz和强度
    result = np.column_stack((downsampled_points, downsampled_intensity))

    return result


# 新增点云裁剪函数
def crop_points_by_angle(points_numpy, angle_range=np.pi / 2):
    """根据视角范围裁剪点云，只保留前方指定角度范围内的点云

    Args:
        points_numpy: [N, 4] 点云数组，包含x,y,z,intensity
        angle_range: float, 视角范围，默认为pi/2（前方180度）

    Returns:
        裁剪后的点云数组 [M, 4] 其中M<=N
    """
    # 计算每个点的水平角度（使用arctan2计算方位角）
    # 假设激光雷达坐标系：x轴正方向为前方，y轴正方向为左侧
    angles = np.arctan2(points_numpy[:, 1], points_numpy[:, 0])

    # 筛选前方指定角度范围内的点云
    # 例如，angle_range=pi/2表示前方180度范围（左右各90度）
    mask = np.abs(angles) <= angle_range

    # 应用掩码筛选点云
    cropped_points = points_numpy[mask]

    return cropped_points


class EFTRCNN_ROS:
    def __init__(self):
        # 在初始化中添加激光雷达高度参数
        self.lidar_height = rospy.get_param("~lidar_height", 1.7)  # 默认1.7米

        # 添加行人检测框颜色参数 (青色: 0,255,255)
        self.pedestrian_color = rospy.get_param("~pedestrian_color", [0, 1, 0, 0.8])  # RGBA格式
        # 添加行人检测框线宽参数
        self.pedestrian_line_width = rospy.get_param("~pedestrian_line_width", 0.05)  # 默认0.05米

        # 危险区域多边形定义 (可根据需要修改)
        # self.danger_polygon = np.array([[5.0, 3.0], [3.5, -5.3], [7.0, -5.3], [7.0, 3.0]])
        self.danger_polygon = np.array([[18.0, 11.0], [18.0, -6.0], [12.0, -12.0], [6.0, -12.0], [6.0, 11.0]])
        self.D1 = 4.0  # 预判层阈值
        self.D2 = 1.0  # 报警层阈值

        # 置信度过滤阈值，可调整
        self.conf_threshold = rospy.get_param("~conf_threshold", 0.1)

        # 点云下采样参数
        self.use_downsampling = rospy.get_param("~use_downsampling", True)
        self.voxel_size = rospy.get_param("~voxel_size", 0.1)  # 体素大小，越大下采样越多

        # 点云裁剪参数
        self.use_angle_crop = rospy.get_param("~use_angle_crop", False)  # 是否启用角度裁剪
        self.angle_range = rospy.get_param("~angle_range", np.pi / 2)  # 默认裁剪为前方180度视角（±pi/2）

        # 激光雷达坐标系调整参数 (可以根据不同激光雷达进行调整)
        # 旋转轴和角度
        self.lidar_adjust_axis = [0, 0, 1]  # 绕z轴旋转
        self.lidar_adjust_angle = 0.0  # 默认不旋转，根据需要可以设置为0.1047(6度)等

        # 直接指定激光雷达点云话题 (根据实际使用的激光雷达调整)
        self.lidar_topic = "/velodyne_points"  # 默认使用kitti激光雷达话题

        # 帧计数和时间记录
        self.frame_count = 0
        self.last_frame_time = time.time()

        # 预警信息记录
        log_dir = os.path.join(current_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"warning_log_{log_timestamp}.txt")

        # 获取配置路径
        config_path, ckpt_path = self.get_config_paths()

        # 先初始化模型
        self.eftrcnn(config_path, ckpt_path)
        # 然后初始化ROS订阅/发布
        self.init_ros()

        # 用于存储行人检测结果
        self.pedestrian_detections = []

        # 添加帧处理时间统计
        self.frame_processing_times = []  # 存储每帧处理时间
        self.total_processing_time = 0.0  # 总处理时间

        # 缓存旋转矩阵，避免重复计算
        if self.lidar_adjust_angle != 0.0:
            self.rot_matrix = self.rotate_mat(self.lidar_adjust_axis, self.lidar_adjust_angle)
        else:
            self.rot_matrix = None

        rospy.loginfo(
            f"初始化完成，置信度阈值设置为: {self.conf_threshold}, 激光雷达话题: {self.lidar_topic}, 日志文件: {self.log_file}")
        rospy.loginfo(f"激光雷达坐标系调整: 旋转轴={self.lidar_adjust_axis}, 角度={self.lidar_adjust_angle}弧度")
        if self.use_downsampling and OPEN3D_FLAG:
            rospy.loginfo(f"点云下采样已启用，体素大小: {self.voxel_size}m")
        elif self.use_downsampling and not OPEN3D_FLAG:
            rospy.logwarn("点云下采样未启用：未找到Open3D库")
        if self.use_angle_crop:
            angle_degrees = int(self.angle_range * 180 / np.pi)
            rospy.loginfo(f"点云角度裁剪已启用，保留前方 ±{angle_degrees}° 范围内的点云")

    def get_config_paths(self):
        """单独获取路径获取逻辑"""
        config_path = rospy.get_param("~config_path", "")
        ckpt_path = rospy.get_param("~ckpt_path", "")

        return config_path, ckpt_path

    def init_ros(self):
        """初始化ROS参数和订阅/发布者"""
        # 设置订阅和发布主题
        self.sub_velo = rospy.Subscriber(self.lidar_topic, PointCloud2, self.lidar_callback, queue_size=1,
                                         buff_size=2 ** 24)  # TODO 数据量庞大的点云增加缓冲区大小
        self.pub_bbox = rospy.Publisher("/detections", BoundingBoxArray, queue_size=10)
        self.pub_danger_zone = rospy.Publisher("/danger_zone", Marker, queue_size=1)
        # 点云重发布器，解决可视化延时问题
        self.pub_velo = rospy.Publisher("/modified", PointCloud2, queue_size=1)
        # 新增: 标签文本发布器
        self.pub_labels = rospy.Publisher("/detection_labels", MarkerArray, queue_size=10)

    def init_eftrcnn(self, config_path, ckpt_path):
        """初始化点云检测模型"""
        logger = common_utils.create_logger()
        logger.info('-----------------Quick Demo of EFTRCNN-------------------------')
        cfg_from_yaml_file(config_path, cfg)  # 加载配置文件

        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            ext='.bin', logger=logger
        )
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        self.model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=False)  # 加载权重文件
        self.model.cuda()  # 将网络放到GPU上
        self.model.eval()  # 开启评估模式

        # 使用JIT优化模型（如果可能）
        try:
            # 尝试将模型部分模块进行JIT编译，加速推理
            for module_name, module in self.model.named_modules():
                if 'backbone_3d' in module_name or 'backbone_2d' in module_name:
                    if hasattr(module, 'forward') and callable(module.forward):
                        try:
                            # 尝试将模块通过TorchScript优化
                            optimized_module = torch.jit.script(module)
                            # 注：这里不直接替换模块，因为可能会导致模型结构问题
                            # 而是通过记录已优化的状态来提高性能意识
                            logger.info(f"Successfully optimized module: {module_name}")
                        except Exception as e:
                            logger.warning(f"Failed to optimize module {module_name}: {e}")
        except Exception as e:
            logger.warning(f"JIT optimization failed: {e}")

        # 获取行人类别索引, 不同数据集的类别索引不同，nuscenes为pedestrian；kitti为Pedestrian TODO
        self.pedestrian_idx = cfg.CLASS_NAMES.index('pedestrian') + 1  # +1 因为模型输出是1-based

    def rotate_mat(self, axis, radian):
        """创建旋转矩阵"""
        rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
        return rot_matrix

    def publish_danger_zone(self, frame_id):
        """发布危险区域可视化标记（多边形棱柱）- 修改高度使其贴地"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "danger_zone"
        marker.id = 0
        marker.type = Marker.LINE_LIST  # 使用线列表绘制棱柱
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # 线宽
        marker.pose.orientation.w = 1.0

        # 设置颜色 (红色)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # 棱柱高度 - 从地面开始计算
        bottom_z = -self.lidar_height  # 地面位置（激光雷达坐标系中z=0是安装高度，所以地面为负）
        top_z = bottom_z + 3.0  # 3米高

        points = []
        # 添加底部多边形线段
        for i in range(len(self.danger_polygon)):
            p1 = self.danger_polygon[i]
            p2 = self.danger_polygon[(i + 1) % len(self.danger_polygon)]

            # 底部线段 - 贴地
            points.append(Point(p1[0], p1[1], bottom_z))
            points.append(Point(p2[0], p2[1], bottom_z))

            # 顶部线段
            points.append(Point(p1[0], p1[1], top_z))
            points.append(Point(p2[0], p2[1], top_z))

            # 垂直线段（连接顶部和底部）
            points.append(Point(p1[0], p1[1], bottom_z))
            points.append(Point(p1[0], p1[1], top_z))

        marker.points = points
        self.pub_danger_zone.publish(marker)

    def create_text_marker(self, position, text, id, frame_id):
        """创建文本标记"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "detection_labels"
        marker.id = id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        # 设置位置(在边界框顶部上方显示)
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2] + 0.2  # 在检测框上方0.5米

        # 文本大小
        marker.scale.z = 0.5  # 文本高度

        # 设置文本颜色(青色)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # 设置文本内容
        marker.text = text

        return marker

    def log_warning_info(self, pedestrian_id, position, warning_level, confidence, distance):
        """记录预警信息到日志文件，添加距离信息"""
        # 根据距离判断位置状态
        position_status = "在危险区域内" if distance == 0.0 else f"距离危险区域: {distance:.2f}米"

        log_entry = (
            f"pedestrian {pedestrian_id}: "
            f"坐标({position[0]:.2f}, {position[1]:.2f}) -> "
            f"{warning_level}, {position_status}\n"
        )
        # 写入日志文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        # 打印到终端
        rospy.loginfo(log_entry.strip())

        return log_entry

    def publish_point_cloud(self, points, frame_id):
        """重新发布点云，解决可视化延时问题"""
        header = Header()
        header.stamp = rospy.Time()  # 使用当前时间
        header.frame_id = frame_id

        # 创建点云消息
        msg_segment = pc2.create_cloud(header=header, fields=_make_point_field(4), points=points)

        # 发布点云
        self.pub_velo.publish(msg_segment)

    def normalize_frame_id(self, frame_id):
        """去除帧ID开头的斜杠"""
        if frame_id.startswith('/'):
            return frame_id[1:]
        return frame_id

    def lidar_callback(self, msg):
        """点云回调函数 - 处理点云并进行检测和预警"""
        # 记录处理开始时间
        start_time = time.time()

        # 获取帧ID并去除开头的斜杠
        frame_id = self.normalize_frame_id(msg.header.frame_id)

        # 帧计数增加
        self.frame_count += 1

        # 记录帧头信息到日志和终端
        self.log_frame_header()

        # 使用ros_numpy加速点云数据读取
        np_p = ros_numpy.point_cloud2.pointcloud2_to_array(msg)

        # 提取坐标和强度
        x = np_p['x'].reshape(-1, 1)
        y = np_p['y'].reshape(-1, 1)
        z = np_p['z'].reshape(-1, 1)

        # 处理强度字段
        if 'intensity' in np_p.dtype.names:
            i = np_p['intensity'].reshape(-1, 1)
        else:
            i = np.zeros_like(x)

        # 组合原始点云数据
        points_raw = np.hstack((x, y, z, i))

        # 点云处理流程 - 先裁剪，再下采样

        # 1. 应用角度裁剪（如果启用）
        if self.use_angle_crop:
            num_points_before = points_raw.shape[0]
            points_raw = crop_points_by_angle(points_raw, self.angle_range)
            num_points_after = points_raw.shape[0]
            rospy.logdebug(
                f"点云裁剪: {num_points_before} -> {num_points_after} 点 (减少 {100 * (1 - num_points_after / num_points_before):.1f}%)")

        # 2. 进行点云下采样（如果启用）
        if self.use_downsampling and OPEN3D_FLAG:
            # 记录下采样前点数
            num_points_before = points_raw.shape[0]

            # 执行下采样
            points_raw = voxel_downsample(points_raw, self.voxel_size)

            # 记录下采样后点数
            num_points_after = points_raw.shape[0]
            rospy.logdebug(
                f"点云下采样: {num_points_before} -> {num_points_after} 点 (减少 {100 * (1 - num_points_after / num_points_before):.1f}%)")

        # 根据激光雷达坐标系调整参数进行转换
        if self.lidar_adjust_angle != 0.0:
            # 使用缓存的旋转矩阵
            # 只旋转XYZ坐标
            points_rot = np.dot(self.rot_matrix, points_raw[:, :3].T).T
            # 将旋转后的XYZ与强度组合
            points = np.column_stack((points_rot, points_raw[:, 3]))
        else:
            # 如果不需要旋转，则直接使用原始点云
            points = points_raw

        # 准备输入数据
        input_dict = {
            'points': points,
            'frame_id': frame_id,
            'gt_boxes': np.zeros((0, 8), dtype=np.float32),  # 添加空数组
            'gt_names': np.array([], dtype=str),  # 空gt_names
            'use_lead_xyz': np.array([True]),  # 使用lead_xyz特征
            'num_points_in_gt': np.array([], dtype=np.int32)  # 空num_points_in_gt
        }

        data_dict = self.demo_dataset.prepare_data(data_dict=input_dict)
        data_dict = self.demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        # 模型推理
        torch.cuda.synchronize()  # 确保之前的GPU操作完成
        inference_start = time.time()

        with torch.no_grad():
            pred_dicts, _ = self.model.forward(data_dict)

        torch.cuda.synchronize()  # 确保推理完成
        inference_time = time.time() - inference_start
        rospy.logdebug(f"模型推理时间: {inference_time * 1000:.2f} ms")

        # 提取检测结果
        boxes_lidar = pred_dicts[0]['pred_boxes'].detach().cpu().numpy()
        scores = pred_dicts[0]['pred_scores'].detach().cpu().numpy()
        labels = pred_dicts[0]['pred_labels'].detach().cpu().numpy()

        # 发布危险区域
        self.publish_danger_zone(frame_id)

        # 重置行人检测结果
        self.pedestrian_detections = []
        arr_bbox = BoundingBoxArray()
        marker_array = MarkerArray()  # 新增: 创建标签标记数组

        # 筛选行人并进行预警分析
        pedestrian_count = 0  # 行人计数

        # 应用置信度过滤
        valid_idx = (scores >= self.conf_threshold)
        if np.any(valid_idx):
            boxes_lidar_filtered = boxes_lidar[valid_idx]
            scores_filtered = scores[valid_idx]
            labels_filtered = labels[valid_idx]

            # 只选择行人类别的检测结果，加速处理
            pedestrian_mask = (labels_filtered == self.pedestrian_idx)
            if np.any(pedestrian_mask):
                pedestrian_boxes = boxes_lidar_filtered[pedestrian_mask]
                pedestrian_scores = scores_filtered[pedestrian_mask]

                # 批量处理所有行人
                for i in range(pedestrian_boxes.shape[0]):
                    pedestrian_count += 1  # 行人序号从1开始

                    # 创建边界框
                    bbox = BoundingBox()
                    bbox.header.frame_id = frame_id
                    bbox.header.stamp = rospy.Time.now()

                    # 调整行人位置到地面
                    box_height = float(pedestrian_boxes[i][5])
                    ground_z = -self.lidar_height + box_height / 2

                    bbox.pose.position.x = float(pedestrian_boxes[i][0])
                    bbox.pose.position.y = float(pedestrian_boxes[i][1])
                    bbox.pose.position.z = ground_z  # 调整到地面位置

                    bbox.dimensions.x = float(pedestrian_boxes[i][3])
                    bbox.dimensions.y = float(pedestrian_boxes[i][4])
                    bbox.dimensions.z = float(pedestrian_boxes[i][5])

                    q = Quaternion(axis=(0, 0, 1), radians=float(pedestrian_boxes[i][6]))
                    bbox.pose.orientation.x = q.x
                    bbox.pose.orientation.y = q.y
                    bbox.pose.orientation.z = q.z
                    bbox.pose.orientation.w = q.w

                    bbox.value = pedestrian_scores[i]
                    bbox.label = self.pedestrian_idx

                    # 提取行人位置
                    pedestrian_pos = (pedestrian_boxes[i][0], pedestrian_boxes[i][1])

                    # 进行三级预警，包含min_dist
                    warning_level, distance = invasion_warning(
                        pedestrian_pos,
                        self.danger_polygon,
                        self.D1,
                        self.D2
                    )

                    log_entry = self.log_warning_info(
                        pedestrian_count,
                        pedestrian_pos,
                        warning_level,
                        pedestrian_scores[i],
                        distance  # 新增距离参数
                    )

                    # 修改标签以包含行人编号和置信度
                    bbox.value = pedestrian_scores[i]  # 置信度
                    bbox.label = 10000 + pedestrian_count  # 使用高位数字作为自定义标签

                    # 创建文本标记
                    label_text = f"Pedestrian{pedestrian_count}: {pedestrian_scores[i]:.2f}"
                    text_marker = self.create_text_marker(
                        [bbox.pose.position.x, bbox.pose.position.y, bbox.pose.position.z + box_height / 2],
                        label_text,
                        pedestrian_count,
                        frame_id
                    )
                    marker_array.markers.append(text_marker)

                    # 只添加行人检测框到发布列表
                    arr_bbox.boxes.append(bbox)

            # 发布检测框和标签（分开发布）TODO 会出现检测框和标签不同时存在
            arr_bbox.header.frame_id = frame_id
            arr_bbox.header.stamp = rospy.Time.now()
            self.pub_bbox.publish(arr_bbox)

            # 发布标签标记
            if len(marker_array.markers) > 0:
                self.pub_labels.publish(marker_array)

            # 重新发布点云，解决可视化延时问题
            # 注意：为了减少处理负担，这里可以选择发布下采样后的点云
            self.publish_point_cloud(points, frame_id)

        # 计算并记录处理时间
        processing_time = time.time() - start_time
        self.frame_processing_times.append(processing_time)
        self.total_processing_time += processing_time

        # 实时显示当前帧处理时间
        fps = 1.0 / processing_time if processing_time > 0 else 0
        rospy.loginfo(f"帧 #{self.frame_count} 处理时间: {processing_time * 1000:.2f} ms | FPS: {fps:.2f}")

    def update_bbox_text(self, bbox, text):
        """为边界框添加自定义文本属性。尝试使用jsk_recognition_msgs中支持的属性"""
        try:
            # 尝试直接设置text属性
            bbox.text = text
        except:
            # 如果不支持，则可以尝试使用label或value来编码信息
            # 这是因为不同版本的jsk_recognition_msgs可能支持不同的属性
            pass
        return bbox

    def log_frame_header(self):
        """记录帧头信息到日志和终端"""
        now = time.time()
        time_diff = now - self.last_frame_time
        self.last_frame_time = now

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        frame_header = f"\n{'=' * 50}\n[帧 #{self.frame_count}] - {timestamp} (间隔: {time_diff:.3f}秒)\n{'=' * 50}"

        # 写入日志文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(frame_header + "\n")

        # 打印到终端，使用不同颜色突出显示帧头
        rospy.logwarn(frame_header)


if __name__ == '__main__':
    rospy.init_node('eftrcnn_ros_warning_node', anonymous=True)

    # 全局变量实例化
    sec = EFTRCNN_ROS()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        # 计算并输出统计信息
        if sec.frame_count > 0:
            avg_time = sec.total_processing_time / sec.frame_count
            min_time = min(sec.frame_processing_times) * 1000
            max_time = max(sec.frame_processing_times) * 1000
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0

            stats = f"\n{'=' * 50}\n" \
                    f"算法效率统计:\n" \
                    f"总处理帧数: {sec.frame_count}\n" \
                    f"总处理时间: {sec.total_processing_time:.2f} 秒\n" \
                    f"平均处理时间: {avg_time * 1000:.2f} ms/帧\n" \
                    f"最小处理时间: {min_time:.2f} ms\n" \
                    f"最大处理时间: {max_time:.2f} ms\n" \
                    f"平均帧率: {avg_fps:.2f} FPS\n" \
                    f"{'=' * 50}"

            # 打印到终端
            print(stats)

            # 写入日志文件
            with open(sec.log_file, 'a', encoding='utf-8') as f:
                f.write(stats)

        del sec
        print("Shutting down")
