import mayavi.mlab as mlab
import numpy as np
import torch
from numpy.ma.core import indices

"""
1. 增加标签类别：Pedestrian 1；
2. 增加类别置信分数；
2. 增加行使方向；

"""



box_colormap = [
    # [1, 1, 1], # 背景
    # [0, 1, 0], # 类别1
    # [0, 1, 1], # 类别2
    # [1, 1, 0], # 类别3

    [0, 1, 0], # 类别1 绿色
    [0, 1, 1], # 类别2 青色
    [1, 1, 0], # 类别3 黄色
    [1, 0, 0], # 备用 红色

]


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def visualize_pts(pts, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
                  show_intensity=False, size=(600, 600), draw_origin=True):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    if show_intensity:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)

    return fig


def draw_sphere_pts(pts, color=(0, 1, 0), fig=None, bgcolor=(0, 0, 0), scale_factor=0.2):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()

    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(600, 600))

    if isinstance(color, np.ndarray) and color.shape[0] == 1:
        color = color[0]
        color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    if isinstance(color, np.ndarray):
        pts_color = np.zeros((pts.__len__(), 4), dtype=np.uint8)
        pts_color[:, 0:3] = color
        pts_color[:, 3] = 255
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], np.arange(0, pts_color.__len__()), mode='sphere',
                          scale_factor=scale_factor, figure=fig)
        G.glyph.color_mode = 'color_by_scalar'
        G.glyph.scale_mode = 'scale_by_vector'
        G.module_manager.scalar_lut_manager.lut.table = pts_color
    else:
        mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='sphere', color=color,
                      colormap='gnuplot', scale_factor=scale_factor, figure=fig)

    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
    mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), line_width=3, tube_radius=None, figure=fig)

    return fig


def draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5)):
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    return fig


def draw_multi_grid_range(fig, grid_size=20, bv_range=(-60, -60, 60, 60)):
    for x in range(bv_range[0], bv_range[2], grid_size):
        for y in range(bv_range[1], bv_range[3], grid_size):
            fig = draw_grid(x, y, x + grid_size, y + grid_size, fig)

    return fig


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None, class_names=None):
    if not isinstance(points, np.ndarray):
        points = points.cpu().numpy()
    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = ref_boxes.cpu().numpy()
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes.cpu().numpy()
    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        ref_scores = ref_scores.cpu().numpy()
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.cpu().numpy()

    fig = visualize_pts(points)
    fig = draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))
    if gt_boxes is not None:
        corners3d = boxes_to_corners_3d(gt_boxes)
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)

    # if ref_boxes is not None and len(ref_boxes) > 0:
    #     ref_corners3d = boxes_to_corners_3d(ref_boxes)
    #     if ref_labels is None:
    #         fig = draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), cls=ref_scores, max_num=100)
    #     else:
    #         for k in range(ref_labels.min(), ref_labels.max() + 1):
    #             cur_color = tuple(box_colormap[k % len(box_colormap)])
    #             mask = (ref_labels == k)
    #             fig = draw_corners3d(ref_corners3d[mask], fig=fig, color=cur_color, cls=ref_scores[mask], max_num=100)

    if ref_boxes is not None and len(ref_boxes) > 0:
        ref_corner3d = boxes_to_corners_3d(ref_boxes)
        if ref_labels is None:
            fig = draw_corners3d(ref_corner3d, fig=fig, color=(0, 1, 0), max_num=100)
        else:
            # 创建类别计数器字典，按照置信分数由高到低
            class_counters = {cls_id: 1 for cls_id in np.unique(ref_labels)}

            for k in range(ref_labels.min(), ref_labels.max() + 1):
                cur_color = tuple(box_colormap[k % len(box_colormap)])
                mask = (ref_labels == k)

                if not np.any(mask):
                    continue # 如果没有该类别的框，跳过
                # 获取类别名称,按照置信度排序
                if class_names is not None and k < len(class_names):
                    cls_name = class_names[k]
                else:
                    cls_name = str(k)

                # 生成带序号的标签
                cur_cls = []
                indices = np.where(mask)[0] # 获取当前类别的所有检测框索引

                for idx in indices:
                    # 获取当前实例的计数器值并递增
                    count = class_counters[k]
                    class_counters[k] += 1

                    # 组合标签文本
                    if ref_scores is not None:
                        text = f"{cls_name} {count}\n{ref_scores[idx]:.2f}"
                    else:
                        text = f"{cls_name} {count}"
                    cur_cls.append(text)

                    # 绘制带序号标签的检测框
                fig = draw_corners3d(
                    ref_corner3d[mask], fig=fig,
                    color=cur_color, cls=cur_cls,
                    max_num=100
                )
                # if ref_scores is not None:
                #     for score in ref_scores[mask]:
                #         cur_cls.append(f"{cls_name}\n{score:.2f}")
                # else:
                #     cur_cls = [cls_name for _ in range(mask.sum())]
                # fig = draw_corners3d(ref_corner3d[mask], fig=fig, color=cur_color, cls=cur_cls, max_num=100)

    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    return fig


def draw_corners3d(corners3d, fig, color=(1, 1, 1), line_width=2, cls=None, tag='', max_num=500, tube_radius=None):
    """
    :param corners3d: (N, 8, 3)
    :param fig:
    :param color:
    :param line_width:
    :param cls:
    :param tag:
    :param max_num:
    :return:
    """
    import mayavi.mlab as mlab
    num = min(max_num, len(corners3d))
    for n in range(num):
        b = corners3d[n]  # (8, 3)

        if cls is not None:
            # if isinstance(cls, np.ndarray):
            #     mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%.2f' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)
            # else:
            #     mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%s' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)
            text = cls[n] if isinstance(cls, (list, np.ndarray)) else cls
            lines = str(text).split('\n')
            line_spacing = 0.2  # 行间距
            for i, line in enumerate(lines):
                z_offset = -i * line_spacing  # 在Z轴方向递减
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2] + z_offset, line,
                            scale=(0.2, 0.2, 0.2), color=color, figure=fig)

        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

        i, j = 0, 5
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)
        i, j = 1, 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)

    return fig


def draw_danger_zone(polygon, height=8.0, ground_z=0.0, color=(1, 0, 0), opacity=0.2, fig=None):
    """
    绘制三维危险区域棱柱
    :param polygon: 底面多边形顶点 [[x1,y1], [x2,y2], ...]
    :param height: 棱柱高度（单位：米）
    :param ground_z: 棱柱底面Z坐标(从点云中获取)
    :param color: 颜色 (R,G,B)
    :param opacity: 透明度 0~1
    :param fig: mayavi figure对象
    :return: fig
    """
    from mayavi import mlab
    import numpy as np

    polygon = np.array(polygon)
    n = len(polygon)

    polygon = np.array(polygon)
    n = len(polygon)

    # 创建底面和顶面顶点
    bottom = np.hstack([polygon, np.full((n, 1), ground_z)])  # 底面贴合地面
    top = np.hstack([polygon, np.full((n, 1), ground_z + height)])  # 顶面高度

    # 绘制底面和顶面多边形（保持与地面平行）
    for z_level in [ground_z, ground_z + height]:
        x = np.append(polygon[:, 0], polygon[0, 0])
        y = np.append(polygon[:, 1], polygon[0, 1])
        z = np.full_like(x, z_level)
        mlab.plot3d(x, y, z, color=color, tube_radius=None, line_width=2.0, figure=fig)

    # 绘制侧面
    for i in range(n):
        # 连接底面和顶面的对应顶点
        x = [bottom[i, 0], top[i, 0], top[(i + 1) % n, 0], bottom[(i + 1) % n, 0], bottom[i, 0]]
        y = [bottom[i, 1], top[i, 1], top[(i + 1) % n, 1], bottom[(i + 1) % n, 1], bottom[i, 1]]
        z = [bottom[i, 2], top[i, 2], top[(i + 1) % n, 2], bottom[(i + 1) % n, 2], bottom[i, 2]]
        mlab.plot3d(x, y, z, color=color, tube_radius=None, line_width=2.0, figure=fig)

    # 添加半透明表面
    xx = np.vstack([bottom[:, 0], top[:, 0]])
    yy = np.vstack([bottom[:, 1], top[:, 1]])
    zz = np.vstack([bottom[:, 2], top[:, 2]])
    mlab.mesh(xx, yy, zz, color=color, opacity=opacity, figure=fig)

    return fig