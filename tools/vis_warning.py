import argparse
import glob
from pathlib import Path
import mayavi.mlab as mlab
import time
from av2.map.pedestrian_crossing import PedestrianCrossing
from networkx.algorithms.bipartite.basic import color
from skimage.draw import polygon

from visual_utils import visualize_utils2 as V

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

#------------三级预警模型核心模块----------------
def ray_casting(point, polygon):
    # -------射线法-------
    """
    param point: 人员坐标 [x, y]
    param polygon: 危险区域多边形顶点列表，格式为[[x1, y1], [x2, y2], ...]
    return: 点是否在多边形内（True(内部)/False(外部)）
    """
    x, y = point
    crossings = 0
    polygon = np.array(polygon)
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i + 1) % len(polygon)]
        ax, ay = a
        bx, by = b

        # 检查点是否在边的y范围内
        if (ay > y) != (by > y):
            # 计算射线与边的交点
            x_intersect = (bx - ax) * (y - ay) / (by - ay + 1e-9) + ax
            if x < x_intersect:  # 交点在射线右侧
                crossings += 1

    return crossings % 2 == 1


def point_to_segment_dist(p, a, b):
    # -------点到线段的最短距离-------
    """
    param point: 人员坐标 [x, y]
    param a: 线段起点坐标 [x, y]
    param b: 线段终点坐标 [x, y]
    return: 点到线段的最短距离
    """
    ap = np.array(p) - np.array(a)
    ab = np.array(b) - np.array(a)
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-9)  # 避免除以零

    if t <= 0:
        return np.linalg.norm(ap)
    elif t >= 1:
        return np.linalg.norm(np.array(p) - np.array(b))
    else:
        projection = a + t * ab
    return np.linalg.norm(np.array(p) - projection)


def min_distance_to_polygon(point, polygon):
    # -------计算点到多边形的最短距离-------
    """
    param point: 人员坐标 [x, y]
    param polygon: 危险区域多边形顶点列表，格式为[[x1, y1], [x2, y2], ...]
    return: 点到多边形的最短距离
    """
    min_dist = float('inf')
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i + 1) % len(polygon)]
        dist = point_to_segment_dist(point, a, b)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def invasion_warning(point, polygon, D1=4.0, D2=1.0):
    """
    param point: 人员坐标 [x, y]
    param polygon: 危险区域多边形顶点列表
    param D1: 预判层阈值（默认4米）
    param D2: 报警层阈值（默认1米）
    return: 预警级别和最小距离（'L1', 'L2', 'L3'）, distance
    """
    # 判断点是否在多边形内
    is_inside = ray_casting(point, polygon)
    if is_inside:
        # 在内部时，距离设为0
        return '报警L3', 0.0
    else:
        d = min_distance_to_polygon(point, polygon)
        if d < D2:
            return '报警L3', d
        elif d <= D1:
            return '预警L2', d
        else:
            return '预判L1', d

#------------------------可视化-----------------------
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    #  制定输入文件扩展名，.bin；.npy；.bag
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    # 置信度阈值参数
    parser.add_argument('--thresh', type=float, default=0.2, help='confidence threshold for filtering detection results')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

#----------------------------------主函数-------------------------------
def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    # 新增预警参数配置
    danger_polygon = [[18.0, 11.0], [18.0, -6.0], [12.0, -12.0], [6.0, -12.0], [6.0, 11.0]]  # 自定义危险区域
    warning_output = Path(args.data_path).parent / "warning_results.txt"  # 预警输出路径

    # 固定尺寸参数
    FIXED_BOX_SIZE = [0.8, 0.8, 1.90]  # 单位：米 (dx, dy, dz)

    logger.info('-----------------基于深度学习三维点云： 危险区域人员入侵检测与三级预警-------------------------')

    # 初始化数据集
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )

    logger.info(f'总帧数: \t{len(demo_dataset)}')

    # 初始化时间统计变量
    total_processing_time = 0.0

    # 加载模型配置、权重文件
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    with torch.no_grad(), open(warning_output, 'w') as warn_f:
        for idx, data_dict in enumerate(demo_dataset):
            start_time = time.time()  # 记录帧开始时间

            # 原始检测流程
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # 获取当前帧点云数据
            points = data_dict['points'][:, 1:].cpu().numpy()  # (N,3) 数组
            # 自动计算地面高度（取点云最低点）
            ground_z = np.percentile(points[:, 2], 5)  # 取5%分位数更鲁棒

            # 筛选Pedestrian类别
            pedestrian_idx = cfg.CLASS_NAMES.index('pedestrian')
            logger.info('目标类别: Pedestrian')

            # 创建多重掩码筛选Pedestrian
            pedestrian_mask = (pred_dicts[0]['pred_labels'] == pedestrian_idx + 1)
            score_mask = pred_dicts[0]['pred_scores'] >= args.thresh
            combined_mask = pedestrian_mask & score_mask

            # 过滤非Pedestrian的检测结果
            filtered_boxes = pred_dicts[0]['pred_boxes'][combined_mask]
            filtered_scores = pred_dicts[0]['pred_scores'][combined_mask]
            filtered_labels = pred_dicts[0]['pred_labels'][combined_mask]

            # +++ 覆盖行人框尺寸 +++
            if len(filtered_boxes) > 0:
                fixed_size = torch.tensor(FIXED_BOX_SIZE,
                                          dtype=filtered_boxes.dtype,
                                          device=filtered_boxes.device)
                filtered_boxes[:, 3:6] = fixed_size

            # 预警处理流程（只处理Pedestrian）
            detections = []
            for box in filtered_boxes.cpu().numpy():
                x, y = round(box[0], 5), round(box[1], 5)
                detections.append([x, y])

            # 执行批量预警并写入文件
            for i, point in enumerate(detections, 1):
                level, min_distance = invasion_warning(point, danger_polygon)  # 获取预警级别和距离
                # 格式化距离输出（保留2位小数）
                min_distance_str = f"{min_distance:.2f}"
                line = f"Pedestrian{i}: 坐标({point[0]:.5f}, {point[1]:.5f}) -> {level}, 最小距离: {min_distance_str}m\n"
                warn_f.write(line)
                logger.info(line.strip())

            # 计算帧处理时间
            frame_time = (time.time() - start_time) * 1000  # 转换为毫秒
            total_processing_time += frame_time
            fps = 1000 / frame_time if frame_time > 0 else 0  # 计算FPS

            logger.info(f"帧 {idx + 1}/{len(demo_dataset)} 处理时间: {frame_time:.2f}ms, FPS: {fps:.2f}")

            # 可视化部分（只显示Pedestrian）
            if len(filtered_boxes) > 0:
                fig = V.draw_scenes(
                    points=data_dict['points'][:, 1:],
                    ref_boxes=filtered_boxes,
                    ref_scores=filtered_scores,
                    ref_labels=filtered_labels - 1,
                    class_names=cfg.CLASS_NAMES
                )

                # 添加危险区域可视化
                V.draw_danger_zone(
                    polygon=danger_polygon,
                    height=3.0,
                    ground_z=ground_z,
                    color=(1, 0, 0),
                    opacity=0.2,
                    fig=fig
                )
                mlab.show()
            else:
                logger.info(f'第{idx + 1}帧未检测到行人')

    # 输出总体性能统计
    avg_frame_time = total_processing_time / len(demo_dataset)
    avg_fps = 1000 / avg_frame_time
    logger.info(f"\n===== 性能统计 =====")
    logger.info(f"总帧数: {len(demo_dataset)}")
    logger.info(f"总处理时间: {total_processing_time / 1000:.2f}秒")
    logger.info(f"平均帧处理时间: {avg_frame_time:.2f}ms")
    logger.info(f"平均FPS: {avg_fps:.2f}")

    logger.info('可视化与三级预警结果处理完成，结果见warning_results.txt.')

if __name__ == '__main__':
    main()