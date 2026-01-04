import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import random
from math import pi

# 全局配置（可根据需求调整）
RANSAC_ITERATIONS = 2000  # RANSAC迭代次数
RANSAC_THRESHOLD = 5.0    # RANSAC重投影误差阈值
DESCRIPTOR_PATCH_SIZE = 11  # 简单描述子的patch尺寸（11x11）
BLEND_MODE = "linear"     # 融合方式："average" / "linear"
#harris_params = {"k":0.04, "kernel_size":3, "window_size":5, "threshold":0.01, "sigma":1.0, "grad_mag_threshold":20}
harris_params = {"k":0.04, "kernel_size":3, "window_size":3, "threshold":0.3, "sigma":1.0, "grad_mag_threshold":20}

#图像可视化工具
#角点可视化
def visualize_corners(img1: np.ndarray, img2: np.ndarray, corners1: np.ndarray, corners2: np.ndarray) -> np.ndarray:
    """
    可视化两张图像的Harris角点（img1红，img2蓝）
    
    参数：
    img1, img2: 输入图像（RGB/BGR/灰度图均可）
    corners1: 第一张图像的角点坐标数组，shape=(N,2)，格式为(x,y)
    corners2: 第二张图像的角点坐标数组，shape=(M,2)，格式为(x,y)
    
    返回：
    并排标注了角点的两张图像（BGR格式，np.ndarray）
    """
    # 分别可视化每张图像的角点
    vis1 = visualize_corners_single(img1, corners1, color=(0, 0, 255))  # 红色
    vis2 = visualize_corners_single(img2, corners2, color=(255, 0, 0))  # 蓝色
    
    # 确保两张图像高度一致（如果不一致则调整）
    h1, w1 = vis1.shape[:2]
    h2, w2 = vis2.shape[:2]
    
    if h1 != h2:
        # 调整到相同高度
        new_h = max(h1, h2)
        vis1_resized = cv2.resize(vis1, (w1, new_h)) if h1 != new_h else vis1
        vis2_resized = cv2.resize(vis2, (w2, new_h)) if h2 != new_h else vis2
        result = np.hstack((vis1_resized, vis2_resized))
    else:
        result = np.hstack((vis1, vis2))
    
    # 添加标题
    result = cv2.putText(result, "Image 1 (Red)", (10, 30), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    result = cv2.putText(result, f"Corners: {len(corners1)}", (10, 60), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    result = cv2.putText(result, "Image 2 (Blue)", (w1 + 10, 30), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    result = cv2.putText(result, f"Corners: {len(corners2)}", (w1 + 10, 60), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    
    # 添加分隔线
    result = cv2.line(result, (w1, 0), (w1, result.shape[0]), (255, 255, 255), 2)
    
    return result
def visualize_corners_single(img: np.ndarray, corners: np.ndarray, color: tuple = (0, 0, 255)) -> np.ndarray:
    """
    可视化单张图像的Harris角点
    
    参数：
    img: 输入图像（RGB/BGR/灰度图均可）
    corners: 角点坐标数组，shape=(N,2)，格式为(x,y)
    color: 绘制角点的颜色（OpenCV BGR格式），默认红色(0,0,255)
    
    返回：
    标注了角点的图像（BGR格式，np.ndarray）
    """
    # 转为彩色图（如果是灰度图）
    if len(img.shape) == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()
    
    # 绘制角点
    for corner in corners:
        if isinstance(corner, tuple) and len(corner) == 2:
            x, y = corner
            cv2.circle(vis, (int(x), int(y)), 3, color, -1)
        else:
            print(f"警告：无效的角点格式: {corner}")
    
    return vis
#显示描述符直方图
def visualize_descriptors(img: np.ndarray, corners: np.ndarray, descriptors: np.ndarray):
    """可视化描述符"""
    if len(corners) == 0:
        return
    
    # 创建可视化图像
    vis = img.copy()
    
    # 绘制角点
    for i, (x, y) in enumerate(corners[:5]):  # 只显示前5个
        cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 0), -1)
        
        # 在角点旁边显示描述符索引
        cv2.putText(vis, f"{i}", (int(x)+10, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow("Corners with Descriptors", vis)
    # 保存测试图像
    cv2.imwrite("intermediate/sift_descriptor/indoor/Corners with Descriptors_left.jpg", vis)
    print("测试图像已保存为 intermediate/sift_descriptor/indoor/Corners with Descriptors_left.jpg")

        # 显示描述符直方图
    if len(descriptors) > 0:
        # 创建一个新的图形用于保存
        save_fig = plt.figure(figsize=(10, 8))
        
        for i in range(min(3, len(descriptors))):
            ax = save_fig.add_subplot(min(3, len(descriptors)), 1, i+1)
            ax.bar(range(len(descriptors[i])), descriptors[i])
            ax.set_title(f"Descriptor {i} (corner at {corners[i]})")
            ax.set_xlabel("Bin")
            ax.set_ylabel("Magnitude")
        
        plt.tight_layout()
        # 保存直方图
        histogram_path = "intermediate/sift_descriptor/indoor/descriptor_histograms_left.jpg"
        save_fig.savefig(histogram_path, dpi=150, bbox_inches='tight')
        plt.close(save_fig)
        print(f"描述符直方图已保存为 {histogram_path}")
        
        # 创建用于显示的图形
        fig, axes = plt.subplots(min(3, len(descriptors)), 1, figsize=(10, 8))
        
        # 如果只有一个描述符，axes不是数组
        if min(3, len(descriptors)) == 1:
            axes = [axes]
        
        for i in range(min(3, len(descriptors))):
            ax = axes[i]
            ax.bar(range(len(descriptors[i])), descriptors[i])
            ax.set_title(f"Descriptor {i} (corner at {corners[i]})")
            ax.set_xlabel("Bin")
            ax.set_ylabel("Magnitude")
        
        plt.tight_layout()
        plt.show()
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#匹配结果可视化
def visualize_matches(
    img1: np.ndarray, 
    img2: np.ndarray, 
    matches: List[Tuple[np.ndarray, np.ndarray]],
    max_matches: int = 50
) -> np.ndarray:
    """
    可视化匹配结果
    
    参数：
        img1, img2: 输入图像
        matches: 匹配点对列表
        max_matches: 最多显示的匹配数量
    
    返回：
        可视化图像
    """
    # 确保图像是彩色图
    if len(img1.shape) == 2:
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_color = img1.copy()
        
    if len(img2.shape) == 2:
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2_color = img2.copy()
    
    # 调整到相同高度
    h1, w1 = img1_color.shape[:2]
    h2, w2 = img2_color.shape[:2]
    
    if h1 != h2:
        target_h = max(h1, h2)
        if h1 != target_h:
            img1_color = cv2.resize(img1_color, (w1, target_h))
        if h2 != target_h:
            img2_color = cv2.resize(img2_color, (w2, target_h))
    
    # 拼接图像
    h, w1 = img1_color.shape[:2]
    _, w2 = img2_color.shape[:2]
    result = np.hstack((img1_color, img2_color))
    
    # 随机颜色生成器（为每条线生成不同颜色）
    np.random.seed(42)
    
    # 限制显示的匹配数量
    display_matches = matches[:min(max_matches, len(matches))]
    
    # 绘制匹配线
    for idx, (pt1, pt2) in enumerate(display_matches):
        # 生成随机颜色
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # 转换为整数坐标
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]) + w1, int(pt2[1])  # 右图坐标需要偏移
        
        # 绘制连接线
        cv2.line(result, (x1, y1), (x2, y2), color, 2)
        
        # 绘制端点
        cv2.circle(result, (x1, y1), 5, color, -1)
        cv2.circle(result, (x2, y2), 5, color, -1)
    
    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    result = cv2.putText(result, f"Feature Matches: {len(display_matches)}/{len(matches)}", 
                         (10, 30), font, 0.8, (0, 255, 0), 2)
    
    return result
#RANSAC匹配可视化
def visualize_ransac_matches(
    img1: np.ndarray, 
    img2: np.ndarray,
    matches: List[Tuple[np.ndarray, np.ndarray]],
    inliers: List[bool],
    H: np.ndarray = None,
    save_path: str = None
):
    """
    可视化RANSAC匹配结果（内点绿色，外点红色）
    """
    if len(matches) == 0:
        print("没有匹配点可显示")
        return
    
    # 创建并排显示的图像
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 创建并排画布
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    
    # 分离内点外点
    inlier_matches = []
    outlier_matches = []
    
    for i, (pt1, pt2) in enumerate(matches):
        if i < len(inliers) and inliers[i]:
            inlier_matches.append((pt1, pt2))
        else:
            outlier_matches.append((pt1, pt2))
    
    # 绘制外点（红色） - 先画，避免被内点覆盖
    for pt1, pt2 in outlier_matches:
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0] + w1), int(pt2[1])  # 第二张图偏移w1
        
        # 绘制连线（红色虚线）
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
        
        # 绘制点
        cv2.circle(vis, (x1, y1), 3, (0, 0, 255), -1)
        cv2.circle(vis, (x2, y2), 3, (0, 0, 255), -1)
    
    # 绘制内点（绿色实线）
    for pt1, pt2 in inlier_matches:
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0] + w1), int(pt2[1])
        
        # 绘制连线（绿色实线）
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
        
        # 绘制点
        cv2.circle(vis, (x1, y1), 4, (0, 255, 0), -1)
        cv2.circle(vis, (x2, y2), 4, (0, 255, 0), -1)
    
    # 添加统计信息
    stats_text = f"All matches: {len(matches)}, Inliers: {len(inlier_matches)} ({len(inlier_matches)/len(matches)*100:.1f}%)"
    cv2.putText(vis, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2)
    
    if H is not None:
        h_text = f"H: [{H[0,0]:.3f}, {H[0,1]:.3f}, {H[0,2]:.3f}]"
        cv2.putText(vis, h_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 1)
    
    # 显示图像
    cv2.imshow("RANSAC Matches (Green=Inliers, Red=Outliers)", vis)
    
    if save_path:
        cv2.imwrite(save_path, vis)
        print(f"RANSAC匹配可视化已保存到: {save_path}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return vis
#重投影误差可视化
def visualize_reprojection_errors(
    matches: List[Tuple[np.ndarray, np.ndarray]],
    H: np.ndarray,
    inliers: List[bool],
    img1: np.ndarray = None,
    img2: np.ndarray = None
):
    """
    可视化重投影误差分布
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    # 计算重投影误差
    points1 = np.array([m[0] for m in matches])
    points2 = np.array([m[1] for m in matches])
    errors = compute_reprojection_error(points1, points2, H)
    
    # 分离内点外点误差
    inlier_errors = [errors[i] for i, is_inlier in enumerate(inliers) if is_inlier]
    outlier_errors = [errors[i] for i, is_inlier in enumerate(inliers) if not is_inlier]
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 子图1: 误差直方图
    ax1 = axes[0, 0]
    bins = np.linspace(0, max(errors)+1, 50)
    ax1.hist(inlier_errors, bins=bins, alpha=0.7, label=f'Inliers ({len(inlier_errors)})', 
             color='green', edgecolor='black')
    ax1.hist(outlier_errors, bins=bins, alpha=0.7, label=f'Outliers ({len(outlier_errors)})', 
             color='red', edgecolor='black')
    ax1.axvline(x=RANSAC_THRESHOLD, color='blue', linestyle='--', 
                label=f'Threshold={RANSAC_THRESHOLD}')
    ax1.set_xlabel('Reprojection Error (pixels)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Reprojection Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 误差箱线图
    ax2 = axes[0, 1]
    error_data = [inlier_errors, outlier_errors]
    bp = ax2.boxplot(error_data, labels=['Inliers', 'Outliers'], patch_artist=True)
    
    # 设置颜色
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax2.axhline(y=RANSAC_THRESHOLD, color='blue', linestyle='--', 
                label=f'Threshold={RANSAC_THRESHOLD}')
    ax2.set_ylabel('Reprojection Error (pixels)')
    ax2.set_title('Error Statistics (Boxplot)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 子图3: 内点分布热力图（第一幅图）
    ax3 = axes[1, 0]
    if img1 is not None:
        ax3.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    
    # 绘制内点外点
    inlier_points = [points1[i] for i, is_inlier in enumerate(inliers) if is_inlier]
    outlier_points = [points1[i] for i, is_inlier in enumerate(inliers) if not is_inlier]
    
    if inlier_points:
        inlier_x, inlier_y = zip(*inlier_points)
        ax3.scatter(inlier_x, inlier_y, c='green', s=50, label='Inliers', 
                   alpha=0.7, edgecolors='white')
    
    if outlier_points:
        outlier_x, outlier_y = zip(*outlier_points)
        ax3.scatter(outlier_x, outlier_y, c='red', s=30, label='Outliers', 
                   alpha=0.5, edgecolors='white')
    
    ax3.set_title('Match Point Distribution on Image 1')
    ax3.legend()
    ax3.axis('off')
    
    # 子图4: 第二幅图像上的点分布
    ax4 = axes[1, 1]
    if img2 is not None:
        ax4.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    
    # 绘制第二幅图的点
    inlier_points2 = [points2[i] for i, is_inlier in enumerate(inliers) if is_inlier]
    outlier_points2 = [points2[i] for i, is_inlier in enumerate(inliers) if not is_inlier]
    
    if inlier_points2:
        inlier_x2, inlier_y2 = zip(*inlier_points2)
        ax4.scatter(inlier_x2, inlier_y2, c='green', s=50, label='Inliers', 
                   alpha=0.7, edgecolors='white')
    
    if outlier_points2:
        outlier_x2, outlier_y2 = zip(*outlier_points2)
        ax4.scatter(outlier_x2, outlier_y2, c='red', s=30, label='Outliers', 
                   alpha=0.5, edgecolors='white')
    
    ax4.set_title('Match Point Distribution on Image 2')
    ax4.legend()
    ax4.axis('off')
    
    plt.suptitle(f'RANSAC Results Analysis: {len(inlier_points)} Inliers, {len(outlier_points)} Outliers', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图形
    save_path = "intermediate/ransac/indoor/reprojection_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"重投影误差分析图已保存到: {save_path}")
    
    plt.show()
    
    # 返回误差统计数据
    stats = {
        'total_matches': len(matches),
        'inlier_count': len(inlier_points),
        'outlier_count': len(outlier_points),
        'inlier_ratio': len(inlier_points) / len(matches) if matches else 0,
        'mean_inlier_error': np.mean(inlier_errors) if inlier_errors else 0,
        'max_inlier_error': np.max(inlier_errors) if inlier_errors else 0,
        'mean_outlier_error': np.mean(outlier_errors) if outlier_errors else 0,
        'max_outlier_error': np.max(outlier_errors) if outlier_errors else 0
    }
    
    return stats


#匹配质量评估函数
def evaluate_matches(
    matches: List[Tuple[np.ndarray, np.ndarray]],
    img_shape1: tuple,
    img_shape2: tuple
) -> dict:
    """
    评估匹配质量
    
    返回：
        包含各种统计信息的字典
    """
    if len(matches) == 0:
        return {"num_matches": 0}
    
    h1, w1 = img_shape1[:2]
    h2, w2 = img_shape2[:2]
    
    # 收集所有匹配点的位移向量
    displacements = []
    x_displacements = []
    y_displacements = []
    
    for pt1, pt2 in matches:
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        displacements.append([dx, dy])
        x_displacements.append(dx)
        y_displacements.append(dy)
    
    displacements = np.array(displacements)
    
    # 计算统计信息
    stats = {
        "num_matches": len(matches),
        "avg_displacement": np.mean(displacements, axis=0),
        "std_displacement": np.std(displacements, axis=0),
        "x_displacement_range": (min(x_displacements), max(x_displacements)),
        "y_displacement_range": (min(y_displacements), max(y_displacements)),
        "avg_distance": np.mean(np.linalg.norm(displacements, axis=1))
    }
    
    return stats

# 主函数：全流程全景图拼接
def panorama_stitching(image1, image2, harris_params):
    """
    Stitch two images into a panorama
    
    Parameters:
    image1, image2: Input images (RGB format)
    harris_params: Dictionary containing Harris detector parameters
                  (k, kernel_size, window_size, threshold)
    
    Returns:
    stitched_image: The resulting panorama
    matches_visualization: Visualization of feature matches
    """

    #Step 1: Feature Point Detection 特征点检测
    #Harris角点检测
    corners1, __ = my_harris_corner_detector(image1, **harris_params)
    corners2, __ = my_harris_corner_detector(image2, **harris_params)

    #SIFT-like特征提取
    desc1 = compute_sift_like_descriptors(image1, corners1) 
    desc2 = compute_sift_like_descriptors(image2, corners2)

    #Step 2: Feature Matching 特征匹配
    #使用NCC相似度计算与NNDR特征匹配
    matches = match_features(desc1, desc2, corners1, corners2,ratio_threshold=0.81, ncc_threshold=0.8, cross_check=True)

    #Step 3: Homography Estimation 单应性估计
    H, inliers = estimate_homography_ransac(matches)

    #Step 4: Image Transformation and Blending 图像变换与融合
    #使用单应矩阵H将第二张图像变换到第一张图像的平面上
    #计算拼接后图像的尺寸
    #图像融合
    stitched_image = stitch_images(image2, image1, H, blend_mode=BLEND_MODE)
    
    return stitched_image

#模块函数
#自定义Harris角点检测器
def my_harris_corner_detector(
        image: np.ndarray,
        k: float,
        kernel_size: int,
        window_size: int,
        threshold: float,
        sigma: float,
        grad_mag_threshold: int
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """
    自定义Harris角点检测器
    参数:
        image: 输入图像 （BGR格式，OpenCV默认读取）
        k: Harris响应函数经验常数（0.04~0.06）
        kernel_size: 高斯滤波核大小
        window_size: 非极大值抑制窗口大小
        threshold: 响应值R的比例阈值（0.01~0.05）
        sigma: 高斯模糊标准差
        grad_mag_threshold: 梯度幅值过滤阈值
    
    返回:
        corners: 角点坐标列表 [(x1,y1), (x2,y2), ...]
        R： Harris响应矩阵
    """
    # 步骤1：图像预处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    gray = np.float32(gray)

    # 步骤2：梯度计算+过滤
    I_x = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0, ksize=3)
    I_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=3)
    I_x = np.abs(I_x)
    I_y = np.abs(I_y)
    
    # 梯度幅值过滤
    grad_mag = np.sqrt(I_x**2 + I_y**2)
    grad_mask = grad_mag > grad_mag_threshold

    # 步骤3：构造M矩阵
    I_x2 = I_x ** 2
    I_y2 = I_y ** 2
    I_xy = I_x * I_y
    S_x2 = cv2.GaussianBlur(I_x2, (kernel_size, kernel_size), sigma)
    S_y2 = cv2.GaussianBlur(I_y2, (kernel_size, kernel_size), sigma)
    S_xy = cv2.GaussianBlur(I_xy, (kernel_size, kernel_size), sigma)

    # 步骤4：计算Harris响应值R
    det_M = S_x2 * S_y2 - S_xy ** 2
    trace_M = S_x2 + S_y2
    R = det_M - k * (trace_M) ** 2
    R = R * grad_mask.astype(np.float64)

    # 步骤5：后处理（阈值化+NMS）
    R_abs = np.abs(R)
    R_max = R_abs.max() if R_abs.max() != 0 else 1e-8
    R_threshold = (R > -0.05) & (R_abs > threshold * R_max)

    corners = []
    h, w = R.shape
    pad = window_size // 2
    for y in range(pad, h - pad):
        for x in range(pad, w - pad):
            if R_threshold[y, x]:
                window = R[y-pad:y+pad+1, x-pad:x+pad+1]
                if np.isclose(R[y, x], window.max(), atol=1e-6):
                    corners.append((x, y))
    
    return corners, R
#SIFT-like描述子计算函数
def compute_sift_like_descriptors(img: np.ndarray, corners: np.ndarray,
                                          patch_size: int = 16, num_bins: int = 8) -> np.ndarray:
    # 转换为灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = gray.astype(np.float32)
    
    # 对图像进行镜像填充，确保可以提取完整的patch
    half_patch = patch_size // 2
    padded_gray = cv2.copyMakeBorder(gray, 
                                     half_patch, half_patch, 
                                     half_patch, half_patch, 
                                     cv2.BORDER_REFLECT)

    # 计算梯度（填充后）
    Ix = cv2.Sobel(padded_gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(padded_gray, cv2.CV_32F, 0, 1, ksize=3)
    
    mag = np.sqrt(Ix**2 + Iy**2)
    ang = np.arctan2(Iy, Ix)
    ang = np.where(ang < 0, ang + 2*pi, ang)
    
    descriptors = []
    
    for corner in corners:
        #获取坐标并调整到填充后图像
        x, y = int(corner[0]) + half_patch, int(corner[1]) + half_patch

        # 直接提取局部区域，不再需要边界检查
        mag_patch = mag[y-half_patch:y+half_patch, x-half_patch:x+half_patch]
        ang_patch = ang[y-half_patch:y+half_patch, x-half_patch:x+half_patch]
        
        # 初始化128维描述符（4x4个子区域，每个8方向）
        descriptor = np.zeros((4, 4, num_bins), dtype=np.float32)
        
        # 子区域大小
        cell_size = patch_size // 4
        
        # 为每个子区域计算方向直方图
        for cell_y in range(4):
            for cell_x in range(4):
                # 提取子区域
                y_start = cell_y * cell_size
                y_end = y_start + cell_size
                x_start = cell_x * cell_size
                x_end = x_start + cell_size
                
                cell_mag = mag_patch[y_start:y_end, x_start:x_end]
                cell_ang = ang_patch[y_start:y_end, x_start:x_end]
                
                # 计算子区域内的方向直方图
                hist = np.zeros(num_bins)
                
                # 创建子区域内的坐标网格（用于计算高斯权重）
                y_coords, x_coords = np.mgrid[y_start:y_end, x_start:x_end]
                y_center = y_start + cell_size // 2
                x_center = x_start + cell_size // 2
                distances = np.sqrt((x_coords - x_center)**2 + (y_coords - y_center)**2)
                sigma = cell_size / 2
                gaussian_weights = np.exp(-distances**2 / (2 * sigma**2))
                
                # 遍历子区域内的每个像素
                for i in range(cell_size):
                    for j in range(cell_size):
                        magnitude = cell_mag[i, j]
                        if magnitude < 1e-6:
                            continue
                        
                        angle = cell_ang[i, j]
                        bin_idx = int(angle / (2*pi) * num_bins) % num_bins
                        
                        # 计算插值
                        exact_bin = angle / (2*pi) * num_bins
                        bin_frac = exact_bin - bin_idx
                        
                        # 加权贡献
                        weight = magnitude * gaussian_weights[i, j]
                        
                        # 分配到分箱
                        hist[bin_idx] += weight * (1 - bin_frac)
                        next_bin = (bin_idx + 1) % num_bins
                        hist[next_bin] += weight * bin_frac
                
                descriptor[cell_y, cell_x, :] = hist
        
        # 展平并归一化描述符
        desc_flat = descriptor.flatten()
        
        # L2归一化
        norm = np.linalg.norm(desc_flat)
        if norm > 0:
            desc_flat = desc_flat / norm
        
        # 阈值截断（增强光照不变性）
        desc_flat = np.clip(desc_flat, 0, 0.2)
        
        # 重新归一化
        norm = np.linalg.norm(desc_flat)
        if norm > 0:
            desc_flat = desc_flat / norm
        
        descriptors.append(desc_flat)
    
    return np.array(descriptors)
#特征匹配函数（NCC + NNDR）
def match_features(
    desc1: np.ndarray, 
    desc2: np.ndarray, 
    corners1: np.ndarray,
    corners2: np.ndarray,
    ratio_threshold: float, 
    ncc_threshold: float, 
    cross_check: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    特征匹配（使用NCC + nearest neighbor/distance ratio test）
    
    参数：
        desc1: 第一张图像的特征描述符，形状 (N1, D)
        desc2: 第二张图像的特征描述符，形状 (N2, D)
        corners1: 第一张图像的角点坐标，形状 (N1, 2)
        corners2: 第二张图像的角点坐标，形状 (N2, 2)
        ratio_threshold: 比率测试阈值（默认0.75）
        ncc_threshold: NCC匹配阈值（默认0.6）
        cross_check: 是否进行双向交叉验证（默认True）
    
    返回：
        匹配点对列表 [(pt1, pt2), ...]
        其中pt1, pt2是坐标数组 [x, y]
    """
    # 确保输入维度正确
    assert len(desc1) == len(corners1), "desc1和corners1长度不一致"
    assert len(desc2) == len(corners2), "desc2和corners2长度不一致"
    
    # 如果描述符数量为0，直接返回空列表
    if len(desc1) == 0 or len(desc2) == 0:
        return []
    
    # 1. 计算归一化互相关（NCC）矩阵
    # 由于描述符已经L2归一化，所以NCC就是点积
    ncc_matrix = desc1 @ desc2.T  # 形状: (N1, N2)
    
    # 添加一个小的epsilon防止数值问题
    eps = 1e-8
    ncc_matrix = np.clip(ncc_matrix, -1 + eps, 1 - eps)

    # 2. 最近邻匹配 + 比率测试
    matches = []
    
    for i in range(len(desc1)):
        similarities = ncc_matrix[i]
        sorted_indices = np.argsort(-similarities)
        
        if len(sorted_indices) >= 2:
            best_idx = sorted_indices[0]
            best_similarity = similarities[best_idx]
            
            second_best_idx = sorted_indices[1]
            second_best_similarity = similarities[second_best_idx]
            
            # 使用相似度差异测试
            similarity_diff = best_similarity - second_best_similarity
            
            # 双重条件：
            # 1. 相似度足够高
            # 2. 相似度差异足够大（最近邻明显优于次近邻）
            if (best_similarity > ncc_threshold and 
                similarity_diff > (1.0 - ratio_threshold)):
                matches.append((i, best_idx, best_similarity))
    
    # 3. 交叉验证
    if cross_check and matches:
        cross_verified_matches = []
        
        for match in matches:
            i, j, similarity = match
            
            # 检查desc2中的第j个特征是否也认为desc1中的第i个特征是最佳匹配
            similarities_from_j = ncc_matrix[:, j]  # 第j列，形状: (N1,)
            
            # 找到desc1中与desc2[j]最相似的特征
            best_match_from_j = np.argmax(similarities_from_j)
            best_similarity_from_j = similarities_from_j[best_match_from_j]
            
            # 如果双向匹配一致
            if best_match_from_j == i:
                cross_verified_matches.append(match)
        
        matches = cross_verified_matches
    
    # 4. 转换为坐标点对
    matched_points = []
    for i, j, similarity in matches:
        pt1 = corners1[i]  # 第一张图像中的点
        pt2 = corners2[j]  # 第二张图像中的点
        matched_points.append((pt1, pt2))
    
    return matched_points
#RANSAC单应性矩阵估计
def estimate_homography_ransac(
    matches: List[Tuple[np.ndarray, np.ndarray]],
    iterations: int = RANSAC_ITERATIONS,
    threshold: float = RANSAC_THRESHOLD
) -> Tuple[np.ndarray, List[bool]]:
    """
    RANSAC估计单应性矩阵
    输出：
    - 最优单应性矩阵 H (3x3)
    - inliers: 内点标记列表（True=内点，False=外点）
    
    逻辑：
    1. 将匹配点分离为两个数组
    2. RANSAC循环：
       a. 随机选择4对点
       b. 计算单应性矩阵
       c. 计算所有点的重投影误差
       d. 统计内点数量（误差小于阈值）
    3. 选择内点最多的H作为最佳模型
    4. 使用所有内点通过最小二乘优化H
    5. 返回最优H和内点标记
    """
    # 将匹配点分离为两个数组
    points1 = np.array([m[0] for m in matches])
    points2 = np.array([m[1] for m in matches])
    n_points = len(matches)
    
    if n_points < 4:
        raise ValueError("至少需要4对匹配点进行RANSAC估计")
    
    best_H = None
    best_inliers = None
    best_inlier_count = 0
    
    # RANSAC主循环
    for i in range(iterations):
        # 随机选择4对点（确保不重复）
        indices = random.sample(range(n_points), 4)
        
        # 计算单应性矩阵
        try:
            H = compute_homography(points1[indices], points2[indices])
        except np.linalg.LinAlgError:
            continue  # 如果矩阵求解失败，跳过此次迭代
        
        # 计算所有点的重投影误差
        errors = compute_reprojection_error(points1, points2, H)
        
        # 统计内点（误差小于阈值）
        inliers = errors < threshold
        inlier_count = np.sum(inliers)
        
        # 更新最佳模型
        if inlier_count > best_inlier_count:
            best_H = H.copy()
            best_inliers = inliers.copy()
            best_inlier_count = inlier_count
    
    if best_H is None:
        raise RuntimeError("RANSAC未能找到有效的单应性矩阵")
    
    # 使用所有内点通过最小二乘优化H
    if best_inlier_count >= 4:
        # 提取内点
        inlier_points1 = points1[best_inliers]
        inlier_points2 = points2[best_inliers]
        
        # 使用所有内点重新计算单应性矩阵
        # 构建线性方程组
        A = []
        for i in range(len(inlier_points1)):
            x1, y1 = inlier_points1[i]
            x2, y2 = inlier_points2[i]
            
            A.append([-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2])
            A.append([0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2])
        
        A = np.array(A)
        
        # 使用SVD求解最小二乘问题
        # 我们希望最小化 ||A * h||^2，约束||h||=1
        _, _, Vt = np.linalg.svd(A)
        h = Vt[-1, :]
        
        # 重构为3x3矩阵并归一化
        refined_H = h.reshape(3, 3)
        refined_H = refined_H / refined_H[2, 2]
        
        # 使用优化后的矩阵重新计算内点
        refined_errors = compute_reprojection_error(points1, points2, refined_H)
        final_inliers = refined_errors < threshold
        
        return refined_H, final_inliers.tolist()
    
    return best_H, best_inliers.tolist()
def compute_homography(points1: np.ndarray, points2: np.ndarray) -> Optional[np.ndarray]:
    """
    输入4对匹配点，计算单应性矩阵 H (3x3)
    points1: 图像1的点 (4,2), points2: 图像2的点 (4,2)
    满足：p2 = H * p1 (齐次坐标)
    
    逻辑：
    1. 将点转换为齐次坐标
    2. 构建线性方程组 A * h = 0
    3. 使用SVD求解最小二乘问题
    4. 将解重构为3x3矩阵并归一化
    """
    if len(points1) != 4 or len(points2) != 4:
        raise ValueError("需要恰好4对匹配点来计算单应性矩阵")
    
    # 构建线性方程组 A * h = 0
    A = []
    for i in range(4):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        
        # 每对点贡献两个方程
        A.append([-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2])
        A.append([0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2])
    
    A = np.array(A)
    
    # 使用SVD求解 A * h = 0 的最小二乘解
    # 最小奇异值对应的右奇异向量就是解
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]  # 取最后一个奇异向量
    
    # 重构为3x3矩阵
    H = h.reshape(3, 3)
    
    # 归一化矩阵，使最后一个元素为1
    H = H / H[2, 2]
    
    return H
def compute_reprojection_error(
    points1: np.ndarray, 
    points2: np.ndarray, 
    H: np.ndarray
) -> np.ndarray:
    """
    计算所有匹配点的重投影误差
    输出：每个点的误差数组 (N,)
    
    逻辑：
    1. 将图像1的点转换为齐次坐标
    2. 使用H计算预测的图像2坐标
    3. 与真实坐标比较，计算欧氏距离
    4. 返回每个点的误差
    """
    # 转换为齐次坐标 (N, 3)
    n_points = len(points1)
    points1_homogeneous = np.ones((n_points, 3))
    points1_homogeneous[:, :2] = points1
    
    # 计算预测点
    predicted_points = np.dot(H, points1_homogeneous.T).T
    
    # 归一化齐次坐标
    predicted_points = predicted_points / predicted_points[:, 2].reshape(-1, 1)
    
    # 提取二维坐标
    predicted_points_2d = predicted_points[:, :2]
    
    # 计算欧氏距离误差
    errors = np.linalg.norm(predicted_points_2d - points2, axis=1)
    
    return errors
#图像拼接模块
def stitch_images(img1: np.ndarray, img2: np.ndarray, H: np.ndarray, 
                  blend_mode: str = BLEND_MODE) -> np.ndarray:
    """
    完整的图像拼接流程
    
    步骤：
    1. 计算拼接图像尺寸和偏移矩阵
    2. 创建变换矩阵，将图像2变换到图像1的坐标系
    3. 变换两张图像到拼接画布
    4. 融合图像
    """
    # 步骤1: 计算拼接图像尺寸和偏移矩阵
    stitched_width, stitched_height, translation = get_stitched_size(img1, img2, H)
    print(f"拼接图像尺寸: {stitched_width} x {stitched_height}")
    
    # 步骤2: 创建拼接画布
    stitched_canvas = np.zeros((stitched_height, stitched_width, 3), dtype=img1.dtype)
    
    # 步骤3: 变换图像1到拼接画布
    # 对于图像1，只需要平移（使用偏移矩阵）
    print("变换图像1到拼接画布...")
    img1_warped = warp_image(img1, translation, (stitched_width, stitched_height))
    
    # 步骤4: 变换图像2到拼接画布
    # 对于图像2，需要先应用H变换，再应用平移
    print("变换图像2到拼接画布...")
    combined_transform = np.dot(translation, H)  # 先H变换，再平移
    img2_warped = warp_image(img2, combined_transform, (stitched_width, stitched_height))
    
    # 步骤5: 融合图像
    print(f"使用 {blend_mode} 模式融合图像...")
    result = blend_images(img1_warped, img2_warped, blend_mode)
    
    # 步骤6: 裁剪黑色边界
    print("裁剪黑色边界...")
    result_cropped = crop_black_borders(result)
    
    print("图像拼接完成!")
    return result_cropped

def warp_image(img: np.ndarray, H: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    使用OpenCV的warpPerspective（C++实现，速度极快）
    """
    width, height = target_size
    
    # OpenCV的warpPerspective内部高度优化（使用C++/SIMD指令）
    warped = cv2.warpPerspective(
        img, 
        H, 
        target_size,
        flags=cv2.INTER_LINEAR,  # 双线性插值
        borderMode=cv2.BORDER_CONSTANT,  # 边界用黑色填充
        borderValue=(0, 0, 0)
    )
    
    return warped
def get_stitched_size(img1: np.ndarray, img2: np.ndarray, H: np.ndarray) -> Tuple[int, int]:
    """
    计算拼接图像的尺寸（覆盖两张图的所有像素）
    返回：(width, height, 偏移矩阵)
    
    逻辑：
    1. 计算图像2四个角点经过H变换后的坐标
    2. 结合图像1的角点，计算边界框
    3. 计算需要的最小画布尺寸和偏移量
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 图像1的四个角点
    corners1 = np.array([
        [0, 0, 1],
        [w1-1, 0, 1],
        [w1-1, h1-1, 1],
        [0, h1-1, 1]
    ], dtype=np.float32).T  # 3x4矩阵
    
    # 图像2的四个角点
    corners2 = np.array([
        [0, 0, 1],
        [w2-1, 0, 1],
        [w2-1, h2-1, 1],
        [0, h2-1, 1]
    ], dtype=np.float32).T  # 3x4矩阵
    
    # 将图像2的角点变换到图像1的坐标系
    corners2_transformed = np.dot(H, corners2)
    corners2_transformed = corners2_transformed / corners2_transformed[2, :]
    
    # 将所有角点合并（包括图像1的角点和变换后的图像2角点）
    all_corners = np.hstack([corners1[:2, :], corners2_transformed[:2, :]])
    
    # 计算边界框
    min_x = np.floor(np.min(all_corners[0, :])).astype(int)
    max_x = np.ceil(np.max(all_corners[0, :])).astype(int)
    min_y = np.floor(np.min(all_corners[1, :])).astype(int)
    max_y = np.ceil(np.max(all_corners[1, :])).astype(int)
    
    # 计算画布尺寸
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    
    # 创建偏移矩阵，使得图像1的原点(0,0)映射到画布的(-min_x, -min_y)
    # 这样图像1的像素都会在画布的正区域内
    translation = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return width, height, translation
def blend_images(img1: np.ndarray, img2_warped: np.ndarray, blend_mode: str = BLEND_MODE) -> np.ndarray:
    """
    图像融合
    blend_mode: "average"（简单平均） / "linear"（线性梯度）
    
    逻辑：
    1. 创建与img1相同尺寸的全景图画布
    2. 将img1放置到画布上
    3. 将img2_warped融合到画布上
    4. 根据融合模式处理重叠区域
    """
    # 确保输入图像尺寸相同
    if img1.shape != img2_warped.shape:
        raise ValueError(f"图像尺寸不匹配: img1={img1.shape}, img2_warped={img2_warped.shape}")
    
    # 创建输出图像
    blended = np.zeros_like(img1, dtype=np.float32)
    
    # 创建掩码，标记哪些位置有像素
    mask1 = np.any(img1 > 0, axis=2).astype(np.float32)
    mask2 = np.any(img2_warped > 0, axis=2).astype(np.float32)
    
    # 计算重叠区域
    overlap = mask1 * mask2
    only_img1 = mask1 * (1 - mask2)
    only_img2 = mask2 * (1 - mask1)
    
    if blend_mode == "average":
        # 简单平均融合
        # 非重叠区域
        blended = img1.astype(np.float32) * only_img1[:, :, np.newaxis]
        blended += img2_warped.astype(np.float32) * only_img2[:, :, np.newaxis]
        
        # 重叠区域：平均
        if np.any(overlap):
            overlap_region = (img1.astype(np.float32) + img2_warped.astype(np.float32)) / 2
            blended += overlap_region * overlap[:, :, np.newaxis]
    
    elif blend_mode == "linear":
        # 线性梯度融合
        # 非重叠区域
        blended = img1.astype(np.float32) * only_img1[:, :, np.newaxis]
        blended += img2_warped.astype(np.float32) * only_img2[:, :, np.newaxis]
        
        # 重叠区域：线性渐变权重
        if np.any(overlap):
            # 创建距离图：距离重叠区域边界的距离
            from scipy import ndimage
            
            # 计算到img2边缘的距离（从img2向外）
            distance_to_img2_edge = ndimage.distance_transform_edt(mask2)
            
            # 计算到img1边缘的距离（从img1向外）
            distance_to_img1_edge = ndimage.distance_transform_edt(mask1)
            
            # 确保在重叠区域内计算权重
            overlap_mask = overlap > 0
            total_distance = np.zeros_like(overlap, dtype=np.float32)
            total_distance[overlap_mask] = (distance_to_img1_edge[overlap_mask] + 
                                           distance_to_img2_edge[overlap_mask])
            
            # 计算权重：离img1越近，权重越大
            weights_img1 = np.zeros_like(overlap, dtype=np.float32)
            weights_img1[overlap_mask] = distance_to_img1_edge[overlap_mask] / total_distance[overlap_mask]
            weights_img2 = 1 - weights_img1
            
            # 应用权重
            blended_overlap = (img1.astype(np.float32) * weights_img1[:, :, np.newaxis] + 
                              img2_warped.astype(np.float32) * weights_img2[:, :, np.newaxis])
            blended += blended_overlap * overlap[:, :, np.newaxis]
    
    else:
        raise ValueError(f"不支持的融合模式: {blend_mode}")
    
    # 转换为原始数据类型
    blended = np.clip(blended, 0, 255).astype(img1.dtype)
    
    return blended

#裁剪黑色边界函数
def crop_black_borders(img: np.ndarray) -> np.ndarray:
    """
    裁剪图像的黑色边界
    """
    # 创建掩码，标记非黑色像素
    mask = np.any(img > 10, axis=2)  # 阈值设为10以避免噪声
    
    # 找到非黑色像素的边界
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return img  # 如果整个图像都是黑色，返回原图
    
    # 计算边界
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # 裁剪图像
    cropped = img[y_min:y_max+1, x_min:x_max+1]
    
    return cropped


#实验函数
def test_basic_experiment():
    """基础实验：测试场景图像对，可视化中间结果"""
    # 加载图像
    img1 = cv2.imread("scene_left.jpg")
    img2 = cv2.imread("scene_right.jpg")
    # Harris参数（调低阈值保证足够角点）
    harris_params = {
        "k": 0.04,
        "kernel_size": 3,
        "window_size": 5,
        "threshold": 0.01
    }
    
    # 拼接
    stitched, matches_vis = panorama_stitching(img1, img2, harris_params)
    
    # 可视化中间结果
    # 1. Harris角点
    corners1 = my_harris_corner_detector(img1, **harris_params)
    corners2 = my_harris_corner_detector(img2, **harris_params)
    corners_vis = visualize_corners(img1, img2, corners1, corners2)
    
    # 2. 保存/显示结果
    cv2.imwrite("corners_visualization.jpg", corners_vis)
    cv2.imwrite("matches_visualization.jpg", matches_vis)
    cv2.imwrite("stitched_panorama.jpg", stitched)
    
    # 显示
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(corners_vis, cv2.COLOR_BGR2RGB)), plt.title("Harris Corners")
    plt.subplot(132), plt.imshow(cv2.cvtColor(matches_vis, cv2.COLOR_BGR2RGB)), plt.title("Feature Matches")
    plt.subplot(133), plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)), plt.title("Stitched Panorama")
    plt.show()
def test_harris_params_analysis(img1, img2):
    """分析Harris参数对结果的影响"""
    #window_size(3,5,7,9,11)和threshold(0.01,0.05,0.1,0.2,0.3)
    #指标：角点数量、匹配数量、RANSAC内点数量、重投影误差
    harris_params["window_size"] = 5
    harris_params["threshold"] = 0.01
    corners_left, __ = my_harris_corner_detector(img1,**harris_params)
    corners_right, __ = my_harris_corner_detector(img2,**harris_params)
    print(f"当前使用的参数：window_size={harris_params['window_size']}, threshold={harris_params['threshold']}")
    print(f"左图检测到 {len(corners_left)} 个角点，右图检测到 {len(corners_right)} 个角点。")
    des1 = compute_sift_like_descriptors(img1, corners_left)
    des2 = compute_sift_like_descriptors(img2, corners_right)
    matches = match_features(des1, des2, corners_left, corners_right,
                           ratio_threshold=0.8, ncc_threshold=0.6)
    print(f"找到 {len(matches)} 个匹配点。")
    H, inliers = estimate_homography_ransac(matches)
    print(f"RANSAC内点数量: {np.sum(inliers)}")
    print(f"内点率: {np.sum(inliers)/len(matches):.2f}")
    errors = compute_reprojection_error(
        np.array([m[0] for m in matches]),
        np.array([m[1] for m in matches]),
        H
    )
    print(f"重投影误差: 平均={np.mean(errors):.2f}") 

def opencv_harris_detector(
    image: np.ndarray,
    blockSize: int = 3,
    ksize: int = 3,
    window_size: int = 5,
    k: float = 0.04,
    grad_mag_threshold: int = 25,  
    quantile: float = 0.97       
) -> list[tuple[int, int]]:
    """标准化OpenCV Harris检测器（强制过滤伪角点）"""
    # 预处理（对齐自定义逻辑）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blockSize, blockSize), sigmaX=1.0)
    gray = np.float32(gray)

    # OpenCV基础Harris计算
    R_cv = cv2.cornerHarris(gray, blockSize=blockSize, ksize=ksize, k=k)
    R_cv = R_cv * (R_cv > 0)  # 仅保留R>0的角点

    # 梯度幅值过滤
    I_x_cv = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    I_y_cv = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag_cv = np.sqrt(I_x_cv**2 + I_y_cv**2)
    grad_mask_cv = grad_mag_cv > grad_mag_threshold
    R_cv = R_cv * grad_mask_cv.astype(np.float32)

    # 分位数阈值+NMS
    non_zero_cv = R_cv[R_cv > 0]
    if len(non_zero_cv) == 0:
        return []
    threshold_cv = np.quantile(non_zero_cv, quantile)
    
    kernel = np.ones((window_size, window_size), np.uint8)  
    R_cv_dilate = cv2.dilate(R_cv, kernel)
    corners_mask = (R_cv > threshold_cv) & (R_cv == R_cv_dilate)

    # 坐标转换
    corners_cv = np.argwhere(corners_mask)
    corners_cv = [(x, y) for y, x in corners_cv]

    return corners_cv

def test_opencv_comparison(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    对比自定义Harris检测器与OpenCV内置Harris检测器的效果
    
    对比维度：
    1. 角点数量
    2. 匹配点数量与内点率 (Inlier Ratio)
    3. 最终拼接效果
    """
    print("\n" + "="*60)
    print("实验 4.3: Full Pipeline Comparison with OpenCV")
    print("="*60)

    # --- 辅助内部函数：执行匹配和拼接流水线 ---
    def run_pipeline(name, c1, c2):
        print(f"\n--- Running Pipeline with {name} ---")
        print(f"角点数量: Left={len(c1)}, Right={len(c2)}")
        
        # 提取特征描述符 (保持描述符算法不变，控制变量)
        d1 = compute_sift_like_descriptors(img1, c1)
        d2 = compute_sift_like_descriptors(img2, c2)
        
        # 特征匹配
        matches = match_features(d1, d2, c1, c2, ratio_threshold=0.81, ncc_threshold=0.8, cross_check=True)
        print(f"初始匹配数量: {len(matches)}")
        
        if len(matches) < 4:
            print("匹配点不足，无法计算单应性矩阵")
            return None, 0, 0
            
        # RANSAC 估计
        H, inliers = estimate_homography_ransac(matches)
        inlier_count = np.sum(inliers)
        inlier_ratio = inlier_count / len(matches) if len(matches) > 0 else 0
        print(f"RANSAC 内点数量: {inlier_count}")
        print(f"内点率 (Inlier Ratio): {inlier_ratio:.2%}")
        
        # 拼接
        stitched = stitch_images(img2, img1, H, blend_mode=BLEND_MODE)
        
        return stitched, inlier_count, inlier_ratio

    # 1. 运行自定义实现 (My Harris)
    # 确保参数与OpenCV尽量对齐
    corners1_my, _ = my_harris_corner_detector(img1, **harris_params)
    corners2_my, _ = my_harris_corner_detector(img2, **harris_params)
    
    res_my, count_my, ratio_my = run_pipeline("Custom Harris", corners1_my, corners2_my)

    # 2. 运行 OpenCV 实现
    # 映射参数: kernel_size -> blockSize
    corners1_cv = opencv_harris_detector(img1)
    corners2_cv = opencv_harris_detector(img2)
    
    res_cv, count_cv, ratio_cv = run_pipeline("OpenCV Harris", corners1_cv, corners2_cv)

    # 3. 可视化对比结果
    plt.figure(figsize=(15, 10))
    result = visualize_corners(img1, img1, corners1_my, corners1_cv)

    # 显示自定义结果
    if res_my is not None:
        plt.subplot(2, 1, 1)
        plt.imshow(cv2.cvtColor(res_my, cv2.COLOR_BGR2RGB))
        plt.title(f"Custom Implementation\nInliers: {count_my} (Ratio: {ratio_my:.1%})")
        plt.axis('off')

    # 显示 OpenCV 结果
    if res_cv is not None:
        plt.subplot(2, 1, 2)
        plt.imshow(cv2.cvtColor(res_cv, cv2.COLOR_BGR2RGB))
        plt.title(f"OpenCV Implementation\nInliers: {count_cv} (Ratio: {ratio_cv:.1%})")
        plt.axis('off')

    plt.tight_layout()
    
    save_path = "results/opencv_comparison/indoor/comparison_result.png"
    # 确保目录存在
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"对比结果已保存至: {save_path}")
    plt.show()
    cv2.imwrite("results/opencv_comparison/indoor/corners_comparison.png", result)
    print(f"角点对比结果已保存至: results/opencv_comparison/indoor/corners_comparison.png")

    # 4. 简要分析打印
    print("\n--- 分析结论 ---")
    print("如果OpenCV结果更好，原因可能包括：")
    print("1. OpenCV使用了更优化的Sobel算子和高斯平滑。")
    print("2. OpenCV的角点响应计算使用了并行指令集加速。")
    print("3. 两者的非极大值抑制(NMS)策略细节可能不同。")

def test_robustness(img1: np.ndarray, img2: np.ndarray, harris_params: Dict):
    """
    鲁棒性测试：旋转不变性和尺度不变性
    
    改进功能：
    1. 旋转不再裁剪图像（自动扩大画布）。
    2. 实现暴力多尺度描述符（Multi-scale Descriptor）以解决尺度变化问题。
    """
    import os
    
    # 1. 设置保存路径 (根据当前测试的数据集调整名称，如 'indoor' 或 'outdoor')
    # 这里假设您正在测试 outdoor 数据，如果是 indoor 请手动修改
    save_dir = os.path.join("results", "robustness_tests", "outdoor")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("\n" + "="*60)
    print("实验 4.4: Robustness Testing")
    print(f"结果将保存至: {save_dir}")
    print("="*60)
    
    # --- 改进版辅助函数：无损旋转图像 ---
    def rotate_image_full(image, angle):
        """
        旋转图像并调整画布大小，确保图像不被裁剪。
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 1. 获取旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 2. 计算旋转后的新边界框尺寸
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # 3. 调整旋转矩阵的平移分量，确保图像居中
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # 4. 执行变换
        rotated = cv2.warpAffine(image, M, (new_w, new_h))
        return rotated

    # --- 辅助函数：可视化角点对比 ---
    def visualize_corner_comparison(img_orig, corners_orig, img_trans, corners_trans, title_suffix):
        # 绘制原图角点
        vis_orig = visualize_corners_single(img_orig, corners_orig, color=(0, 255, 0)) # 绿色
        cv2.putText(vis_orig, f"Original ({len(corners_orig)})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 绘制变换图角点
        vis_trans = visualize_corners_single(img_trans, corners_trans, color=(0, 0, 255)) # 红色
        cv2.putText(vis_trans, f"Transformed ({len(corners_trans)})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 调整高度以便拼接
        h1, w1 = vis_orig.shape[:2]
        h2, w2 = vis_trans.shape[:2]
        target_h = max(h1, h2)
        
        if h1 != target_h: 
            vis_orig = cv2.resize(vis_orig, (int(w1 * target_h / h1), target_h))
        if h2 != target_h: 
            vis_trans = cv2.resize(vis_trans, (int(w2 * target_h / h2), target_h))
        
        comparison = np.hstack((vis_orig, vis_trans))
        
        save_path = os.path.join(save_dir, f"corners_compare_{title_suffix}.jpg")
        cv2.imwrite(save_path, comparison)
        return comparison

    # --- 核心算法：多尺度描述符提取 ---
    def compute_multiscale_descriptors(image, corners, scales=[0.8, 1.0, 1.2, 1.5]):
        """
        [Scale Invariance Improvement]
        对每个角点，强制提取多个不同大小的 Patch 生成描述符。
        相当于构建了一个特征金字塔，增加了匹配不同尺度特征的概率。
        
        参数:
            scales: 相对于 base_patch_size (16) 的缩放因子列表
        返回:
            all_descriptors: (N*M, 128) 扩展后的描述符
            all_keypoints: (N*M, 2) 对应的坐标点
        """
        all_descriptors = []
        all_keypoints = [] 
        
        base_patch_size = 16 
        
        for scale in scales:
            # 计算当前尺度的 patch 大小
            current_patch_size = int(base_patch_size * scale)
            
            # 确保 patch size 是偶数且至少为 8
            current_patch_size = max(8, current_patch_size)
            if current_patch_size % 2 != 0: 
                current_patch_size += 1
            
            # 使用现有的描述符函数，传入变化的 patch_size
            # 注意：这里我们复用了您现有的 compute_sift_like_descriptors_4x4_grid 函数
            descs = compute_sift_like_descriptors(
                image, corners, patch_size=current_patch_size
            )
            
            if len(descs) > 0:
                all_descriptors.append(descs)
                # 复制对应的坐标，因为每个坐标现在有了多个描述符
                all_keypoints.append(corners)
        
        if not all_descriptors:
            return np.array([]), np.array([])
            
        # 垂直堆叠所有尺度的结果
        combined_descs = np.vstack(all_descriptors)
        combined_corners = np.vstack(all_keypoints)
        
        return combined_descs, combined_corners

    # --- 测试流程封装 ---
    def evaluate_transformation(trans_name, modified_img2, file_suffix, use_multiscale=False):
        print(f"\nTesting {trans_name} (Multi-scale: {use_multiscale})...")
        
        # 1. 基础角点检测
        c1, _ = my_harris_corner_detector(img1, **harris_params)
        c2_trans, _ = my_harris_corner_detector(modified_img2, **harris_params)
        
        # 记录原图2角点用于对比
        c2_orig, _ = my_harris_corner_detector(img2, **harris_params)
        visualize_corner_comparison(img2, c2_orig, modified_img2, c2_trans, file_suffix)

        if len(c1) == 0 or len(c2_trans) == 0:
            print("角点检测失败。")
            return

        # 2. 描述符提取 (核心差异)
        if use_multiscale:
            print("  -> 启用多尺度描述符策略...")
            # 对两张图都提取多尺度特征。
            # 这样 img1 的 1.0x 特征 可能匹配上 modified_img2 的 1.2x 特征
            d1, p1 = compute_multiscale_descriptors(img1, c1)
            d2, p2 = compute_multiscale_descriptors(modified_img2, c2_trans)
        else:
            print("  -> 使用单尺度描述符...")
            d1 = compute_sift_like_descriptors(img1, c1)
            d2 = compute_sift_like_descriptors(modified_img2, c2_trans)
            p1, p2 = c1, c2_trans
        
        # 3. 特征匹配
        matches = match_features(d1, d2, p1, p2, ratio_threshold=0.85, ncc_threshold=0.6)
        print(f"  匹配数量: {len(matches)}")
        
        if len(matches) < 4:
            print("  匹配点不足，无法拼接。")
            return

        # 4. RANSAC & 拼接
        try:
            H, inliers = estimate_homography_ransac(matches)
            inlier_count = np.sum(inliers)
            print(f"  RANSAC 内点: {inlier_count}")
            
            # 可视化匹配
            vis_match = visualize_matches(img1, modified_img2, matches, max_matches=30)
            cv2.imwrite(os.path.join(save_dir, f"matches_{file_suffix}.jpg"), vis_match)

            # 执行拼接
            # 注意：H 是从 img1 到 modified_img2 的变换 (根据 match_features 的返回顺序)
            # 但 stitch_images 函数通常期望的是把 img2 变换到 img1
            # 这里我们依然调用 stitch_images(modified_img2, img1, H)
            stitched_img = stitch_images(modified_img2, img1, H, blend_mode="average")
            stitched_img = crop_black_borders(stitched_img)
            
            cv2.imwrite(os.path.join(save_dir, f"stitched_{file_suffix}.jpg"), stitched_img)
            print(f"  拼接成功，结果已保存。")
            
        except Exception as e:
            print(f"  拼接过程出错: {e}")
    # ================= 实验执行 =================
    
    # 1. 旋转测试 (Rotation 30 deg)
    # 使用无损旋转函数
    print("\n>>> 1. 旋转测试 (30度) <<<")
    img2_rotated = rotate_image_full(img2, 30)
    evaluate_transformation("Rotation 30deg", img2_rotated, "rotation_30deg", use_multiscale=False)

    # 2. 缩放测试 (0.8x) - 使用多尺度描述符
    print("\n>>> 2. 缩放测试 (0.8x) <<<")
    h, w = img2.shape[:2]
    img2_scaled = cv2.resize(img2, (int(w * 0.8), int(h * 0.8)))
    evaluate_transformation("Scale 0.8x (Multi-scale)", img2_scaled, "scale_0.8x", use_multiscale=True)

#调试函数
def test_harris_corner_detector():
    """单元测试：Harris角点检测器"""
    image_left = cv2.imread("data/custom/outdoor/left.png")
    image_right = cv2.imread("data/custom/outdoor/right.png")
    if image_left is None or image_right is None:
        print("错误：无法读取图像文件，请检查路径")
        return 
    corners_left, R = my_harris_corner_detector(image_left, **harris_params)
    corners_right, R = my_harris_corner_detector(image_right, **harris_params)
    print(f"左图检测到 {len(corners_left)} 个角点，右图检测到 {len(corners_right)} 个角点。")

    # 可视化角点
    marked_corners = visualize_corners(image_left,image_right, corners_left, corners_right)

    # 显示结果
    cv2.imshow("Corners Detected", marked_corners)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存测试图像
    cv2.imwrite("intermediate/harris_corner/outdoor/test_visualization.jpg", marked_corners)
    print("测试图像已保存为 intermediate/harris_corner/outdoor/test_visualization.jpg")

def test_sift_descriptors():
    """测试SIFT-like描述符计算"""
    # 读取测试图像
    img = cv2.imread("data/custom/indoor/left.jpg")
    if img is None:
        print("错误：无法读取图像")
        return
    corners, _ = my_harris_corner_detector(img, **harris_params)
    if len(corners) == 0:
        print("未检测到角点")
        return
    
    # 选择前10个角点进行测试
    test_corners = corners[:10]
    
    print(f"测试 {len(test_corners)} 个角点的描述符计算")
    #测试描述符
    print("简化SIFT描述符（128维，4x4子区域）:")
    descs = compute_sift_like_descriptors(img, test_corners)
    print(f"   描述符形状: {descs.shape}")
    
    # 可视化描述符
    visualize_descriptors(img, test_corners, descs)
    
def test_feature_matching(img1=None, img2=None):
    """测试特征匹配"""
    
    # 检测角点
    corners1, _ = my_harris_corner_detector(img1, **harris_params)
    corners2, _ = my_harris_corner_detector(img2, **harris_params)
    
    print(f"左图角点: {len(corners1)} 个")
    print(f"右图角点: {len(corners2)} 个")
    
    # 计算描述符
    desc1 = compute_sift_like_descriptors(img1, corners1)
    desc2 = compute_sift_like_descriptors(img2, corners2)
    
    print(f"左图描述符: {desc1.shape}")
    print(f"右图描述符: {desc2.shape}")
    
    # 匹配特征
    matches = match_features(desc1, desc2, corners1, corners2, 
                           ratio_threshold=0.8, ncc_threshold=0.6)
    
    print(f"找到 {len(matches)} 个匹配")
    
    # 评估匹配质量
    stats = evaluate_matches(matches, img1.shape, img2.shape)
    print(f"匹配统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 可视化匹配
    if len(matches) > 0:
        matched_img = visualize_matches(img1, img2, matches, max_matches=15)
        
        cv2.imshow("Feature Matches", matched_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        """
        # 保存结果
        cv2.imwrite("intermediate/feature_match/indoor/feature_matches.jpg", matched_img)
        print(f"匹配结果已保存为 intermediate/feature_match/indoor/feature_matches.jpg")
        """
    
    return matches

def test_ransac(img1: np.ndarray = None, img2: np.ndarray = None):
    matches = test_feature_matching(img1, img2)
    H, inliers = estimate_homography_ransac(matches)
    visualize_ransac_matches(img1, img2, matches, inliers, H, "intermediate/ransac/indoor/ransac_matches.jpg")
    visualize_reprojection_errors(matches, H, inliers, img1, img2)
#测试图像变换与融合模块
def test_image_stitching(img1: np.ndarray = None, img2: np.ndarray = None, save_path: str = None) -> np.ndarray:
    """测试图像拼接功能"""

    matches = test_feature_matching(img1, img2)
    H, __ = estimate_homography_ransac(matches)

    print("=" * 60)
    print("测试图像拼接功能")
    print("=" * 60)
    
    # 测试尺寸计算
    print("\n测试 get_stitched_size 函数...")
    width, height, translation = get_stitched_size(img1, img2, H)
    print(f"拼接尺寸: {width} x {height}")
    print(f"偏移矩阵:\n{translation}")
    
    # 测试图像变换
    print("\n测试 warp_image 函数...")
    target_size = (width, height)
    combined_transform = np.dot(translation, H)
  
   
    img1_warped = warp_image(img1, combined_transform, target_size)
    
    img2_warped = warp_image(img2, translation, target_size)
    
    """
    img1_warped = warp_image(img1, translation, target_size)
    img2_warped = warp_image(img2, combined_transform, target_size)
    """


    # 测试图像融合
    print("\n测试 blend_images 函数...")
    
    # 测试简单平均融合
    print("1. 简单平均融合:")
    blended_average = blend_images(img1_warped, img2_warped, blend_mode="average")
    
    # 测试线性梯度融合
    print("2. 线性梯度融合:")
    blended_linear = blend_images(img1_warped, img2_warped, blend_mode="linear")
      
    # 裁剪黑色边界
    print("3.裁剪黑色边界...")
    result_average_cropped = crop_black_borders(blended_average)
    result_linear_cropped = crop_black_borders(blended_linear)

    print("图像拼接完成!")

    # 可视化结果   
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始图像
    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Image 1 (Original)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Image 2 (Original)')
    axes[0, 1].axis('off')
    
    # 变换后的图像
    axes[0, 2].imshow(cv2.cvtColor(img1_warped, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Image 1 (Warped)')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(img2_warped, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Image 2 (Warped)')
    axes[1, 0].axis('off')
    
    # 融合结果
    axes[1, 1].imshow(cv2.cvtColor(result_linear_cropped, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Linear Blending Result')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cv2.cvtColor(result_average_cropped, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Average Blending Result')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像拼接结果已保存到: {save_path}")
    plt.show()
    
    print("\n测试完成!")
    return result_linear_cropped, result_average_cropped

if __name__ == "__main__":
    img1 = cv2.imread("data/custom/outdoor/left.png")
    img2 = cv2.imread("data/custom/outdoor/right.png")
    #stitched_img = panorama_stitching(img1, img2, harris_params)
    #cv2.imwrite("results/parameter_analysis/outdoor/analyze_window_size/window_size_3.jpg", stitched_img)
    #print("拼接结果已保存到 results/parameter_analysis/outdoor/analyze_window_size/window_size_3.jpg")
  
    #test_image_stitching(img1, img2)

    if img1 is None or img2 is None:
        print("Error: 无法读取图像，请检查路径。")
    else:
        # 直接调用封装好的测试函数
        # 注意：这里会执行函数内部定义好的 旋转测试 和 缩放测试
        test_robustness(img1, img2, harris_params)