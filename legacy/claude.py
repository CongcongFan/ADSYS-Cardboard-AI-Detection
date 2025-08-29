import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import savgol_filter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import cv2
from datetime import datetime
import os

class CardboardWarpDetector:
    def __init__(self, model_path='yolo11l-seg.pt', warp_threshold=0.02):
        """
        初始化纸板弯曲检测器
        
        Args:
            model_path: YOLO模型路径
            warp_threshold: 弯曲度阈值（默认2%）
        """
        self.model = YOLO(model_path)
        self.warp_threshold = warp_threshold
        self.bundle_height = 70  # mm, 一捆纸板的高度
        
    def extract_edges(self, mask_points, gap_threshold=50):
        """
        从分割掩码中提取上下边缘点
        
        Args:
            mask_points: YOLO分割得到的轮廓点
            gap_threshold: 判断断开的阈值（像素）
        
        Returns:
            upper_points: 上边缘点
            lower_points: 下边缘点
        """
        if len(mask_points) == 0:
            return np.array([]), np.array([])
        
        # 方法1：通过轮廓追踪分离上下边缘
        # 找到最左和最右的点
        leftmost_idx = np.argmin(mask_points[:, 0])
        rightmost_idx = np.argmax(mask_points[:, 0])
        
        # 将轮廓分成上下两部分
        # 假设轮廓点是按顺序排列的（YOLO通常返回有序的轮廓点）
        n_points = len(mask_points)
        
        # 处理索引顺序
        if leftmost_idx < rightmost_idx:
            # 上边缘：从左到右
            upper_indices = list(range(leftmost_idx, rightmost_idx + 1))
            # 下边缘：从右到左再到开始
            lower_indices = list(range(rightmost_idx, n_points)) + list(range(0, leftmost_idx + 1))
        else:
            # 上边缘：从左到结尾再到右
            upper_indices = list(range(leftmost_idx, n_points)) + list(range(0, rightmost_idx + 1))
            # 下边缘：从右到左
            lower_indices = list(range(rightmost_idx, leftmost_idx + 1))
        
        # 提取上下边缘点
        upper_points_raw = mask_points[upper_indices]
        lower_points_raw = mask_points[lower_indices]
        
        # 方法2：如果方法1效果不好，使用基于y坐标的分离
        # 计算中心y坐标
        center_y = np.mean(mask_points[:, 1])
        
        # 按x坐标分组
        x_to_y_upper = {}
        x_to_y_lower = {}
        
        for point in mask_points:
            x = int(point[0])
            y = point[1]
            
            # 根据y坐标相对于中心的位置分类
            if y < center_y:
                if x not in x_to_y_upper:
                    x_to_y_upper[x] = []
                x_to_y_upper[x].append(y)
            else:
                if x not in x_to_y_lower:
                    x_to_y_lower[x] = []
                x_to_y_lower[x].append(y)
        
        # 构建上边缘（对于每个x，取上半部分的最低点）
        upper_points = []
        for x in sorted(set(list(x_to_y_upper.keys()) + list(x_to_y_lower.keys()))):
            if x in x_to_y_upper and len(x_to_y_upper[x]) > 0:
                # 上半部分存在点，取最低的（最接近中心）
                upper_points.append([x, max(x_to_y_upper[x])])
            elif x in x_to_y_lower and len(x_to_y_lower[x]) > 0:
                # 只有下半部分有点，取最高的
                upper_points.append([x, min(x_to_y_lower[x])])
        
        # 构建下边缘（对于每个x，取下半部分的最高点）
        lower_points = []
        for x in sorted(set(list(x_to_y_upper.keys()) + list(x_to_y_lower.keys()))):
            if x in x_to_y_lower and len(x_to_y_lower[x]) > 0:
                # 下半部分存在点，取最高的（最接近中心）
                lower_points.append([x, min(x_to_y_lower[x])])
            elif x in x_to_y_upper and len(x_to_y_upper[x]) > 0:
                # 只有上半部分有点，取最低的
                lower_points.append([x, max(x_to_y_upper[x])])
        
        # 如果上下边缘有交叉，使用更简单的方法
        if len(upper_points) > 0 and len(lower_points) > 0:
            upper_points = np.array(upper_points)
            lower_points = np.array(lower_points)
            
            # 检查是否有交叉
            avg_upper_y = np.mean(upper_points[:, 1])
            avg_lower_y = np.mean(lower_points[:, 1])
            
            if avg_upper_y > avg_lower_y:  # 如果上边缘在下边缘下面，说明有问题
                # 使用最简单的方法：对每个x取最小和最大y
                x_to_y = {}
                for point in mask_points:
                    x = int(point[0])
                    y = point[1]
                    if x not in x_to_y:
                        x_to_y[x] = []
                    x_to_y[x].append(y)
                
                upper_points = []
                lower_points = []
                
                for x in sorted(x_to_y.keys()):
                    y_values = x_to_y[x]
                    if len(y_values) > 0:
                        upper_points.append([x, min(y_values)])  # 上边缘（y值较小）
                        lower_points.append([x, max(y_values)])  # 下边缘（y值较大）
                
                upper_points = np.array(upper_points)
                lower_points = np.array(lower_points)
        else:
            upper_points = np.array(upper_points) if len(upper_points) > 0 else np.array([])
            lower_points = np.array(lower_points) if len(lower_points) > 0 else np.array([])
        
        # 处理断开的部分（插值）
        if len(upper_points) > 0:
            upper_points = self.handle_gaps(upper_points, gap_threshold)
        if len(lower_points) > 0:
            lower_points = self.handle_gaps(lower_points, gap_threshold)
        
        return upper_points, lower_points
    
    def handle_gaps(self, points, gap_threshold):
        """
        处理边缘中的断开部分，使用插值填充
        
        Args:
            points: 边缘点
            gap_threshold: 判断断开的阈值
        
        Returns:
            filled_points: 填充后的点
        """
        if len(points) < 2:
            return points
        
        filled_points = []
        for i in range(len(points) - 1):
            filled_points.append(points[i])
            
            # 检查是否有断开
            x_gap = points[i + 1][0] - points[i][0]
            if x_gap > gap_threshold:
                # 使用线性插值填充
                num_fill = int(x_gap / 10)  # 每10像素插入一个点
                x_fill = np.linspace(points[i][0], points[i + 1][0], num_fill + 2)[1:-1]
                y_fill = np.interp(x_fill, 
                                 [points[i][0], points[i + 1][0]], 
                                 [points[i][1], points[i + 1][1]])
                for x, y in zip(x_fill, y_fill):
                    filled_points.append([x, y])
        
        filled_points.append(points[-1])
        return np.array(filled_points)
    
    def fit_curve(self, points, method='polynomial', degree=3):
        """
        拟合曲线
        
        Args:
            points: 边缘点
            method: 拟合方法 ('polynomial', 'spline', 'fourier')
            degree: 多项式度数或样条平滑度
        
        Returns:
            fitted_curve: 拟合的曲线函数
            params: 拟合参数
        """
        if len(points) < degree + 1:
            return None, None
        
        x = points[:, 0]
        y = points[:, 1]
        
        if method == 'polynomial':
            # 多项式拟合
            coeffs = np.polyfit(x, y, degree)
            fitted_curve = np.poly1d(coeffs)
            return fitted_curve, coeffs
            
        elif method == 'spline':
            # 样条插值
            # 先平滑数据
            if len(x) > 5:
                y_smooth = savgol_filter(y, min(len(y), 51), min(3, len(y)-1))
            else:
                y_smooth = y
            
            # 创建样条
            spline = UnivariateSpline(x, y_smooth, s=degree*len(x))
            return spline, None
            
        elif method == 'fourier':
            # 简化的傅里叶拟合（使用多项式近似）
            # 归一化x到[0, 2π]
            x_norm = (x - x.min()) / (x.max() - x.min()) * 2 * np.pi
            
            # 使用前几个傅里叶分量
            n_components = min(5, len(x) // 10)
            A = []
            for k in range(n_components):
                A.append(np.cos(k * x_norm))
                if k > 0:
                    A.append(np.sin(k * x_norm))
            
            A = np.column_stack(A)
            coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
            
            def fourier_curve(x_eval):
                x_eval_norm = (x_eval - x.min()) / (x.max() - x.min()) * 2 * np.pi
                result = coeffs[0] * np.ones_like(x_eval)
                idx = 1
                for k in range(1, n_components):
                    result += coeffs[idx] * np.cos(k * x_eval_norm)
                    idx += 1
                    result += coeffs[idx] * np.sin(k * x_eval_norm)
                    idx += 1
                return result
            
            return fourier_curve, coeffs
        
        return None, None
    
    def calculate_warp(self, upper_curve, lower_curve, x_range, image_width):
        """
        计算弯曲度
        
        Args:
            upper_curve: 上边缘拟合曲线
            lower_curve: 下边缘拟合曲线
            x_range: x坐标范围
            image_width: 图像宽度
        
        Returns:
            warp_value: 弯曲度值
            warp_type: 弯曲类型（'convex', 'concave', 'straight', 's-curve'）
        """
        if upper_curve is None or lower_curve is None:
            return 0, 'unknown'
        
        # 在x范围内采样
        x_sample = np.linspace(x_range[0], x_range[1], 200)
        
        # 计算上下边缘的y值
        y_upper = upper_curve(x_sample)
        y_lower = lower_curve(x_sample)
        
        # 计算中心线
        y_center = (y_upper + y_lower) / 2
        
        # 计算平均厚度
        thicknesses = y_lower - y_upper
        avg_thickness = np.mean(thicknesses)
        
        # 方法1：基于中心线的偏差
        # 拟合一条直线到中心线
        linear_fit = np.polyfit(x_sample, y_center, 1)
        y_linear = np.polyval(linear_fit, x_sample)
        
        # 计算偏差
        deviations = y_center - y_linear
        max_deviation = np.max(np.abs(deviations))
        
        # 方法2：基于厚度变化
        # 理想情况下，厚度应该是恒定的
        thickness_std = np.std(thicknesses)
        thickness_variation = thickness_std / avg_thickness if avg_thickness > 0 else 0
        
        # 方法3：基于曲率
        # 计算二阶导数（曲率的近似）
        curvature_upper = np.gradient(np.gradient(y_upper))
        curvature_lower = np.gradient(np.gradient(y_lower))
        curvature_center = np.gradient(np.gradient(y_center))
        
        # 计算最大曲率
        max_curvature = np.max(np.abs(curvature_center))
        
        # 综合弯曲度计算
        # 主要基于中心线偏差，考虑长度归一化
        length = x_range[1] - x_range[0]
        
        # 弯曲度 = 最大偏差 / 纸板长度
        warp_ratio_deviation = max_deviation / length if length > 0 else 0
        
        # 也考虑厚度变化（权重较小）
        warp_ratio_thickness = thickness_variation * 0.1
        
        # 最终弯曲度
        warp_ratio = warp_ratio_deviation + warp_ratio_thickness
        
        # 判断弯曲类型
        # 检查是否有S形弯曲
        sign_changes = np.sum(np.diff(np.sign(deviations)) != 0)
        
        if warp_ratio < 0.005:
            warp_type = 'straight'
        elif sign_changes >= 2:
            warp_type = 's-curve'  # S形弯曲
        else:
            # 判断凸起还是凹陷
            # 取中间60%的区域来判断主要趋势
            start_idx = int(len(deviations) * 0.2)
            end_idx = int(len(deviations) * 0.8)
            middle_deviation = np.mean(deviations[start_idx:end_idx])
            
            if middle_deviation > 0:
                warp_type = 'convex'  # 凸起（中心线向下偏移）
            else:
                warp_type = 'concave'  # 凹陷（中心线向上偏移）
        
        # 输出调试信息
        print(f"  最大偏差: {max_deviation:.2f} 像素")
        print(f"  纸板长度: {length:.2f} 像素")
        print(f"  厚度变化: {thickness_variation:.4f}")
        print(f"  最大曲率: {max_curvature:.6f}")
        print(f"  弯曲度: {warp_ratio:.6f}")
        
        return warp_ratio, warp_type
    
    def detect_warp(self, image_path, save_results=True, output_dir='results', target_class=None):
        """
        主检测函数
        
        Args:
            image_path: 图像路径
            save_results: 是否保存结果
            output_dir: 输出目录
            target_class: 目标类别索引（如果知道纸板的类别）
        
        Returns:
            results: 检测结果字典
        """
        # 创建输出目录
        if save_results and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # YOLO检测
        print(f"正在处理图像: {image_path}")
        results = self.model.predict(image_path, conf=0.25)  # 添加置信度阈值
        
        if len(results) == 0 or results[0].masks is None:
            print("未检测到纸板")
            return None
        
        # 选择最合适的纸板目标
        # 策略：选择最大的、最像矩形的物体
        best_idx = 0
        best_score = 0
        
        for idx in range(len(results[0].masks.xy)):
            mask_points = results[0].masks.xy[idx]
            if len(mask_points) < 4:
                continue
            
            # 计算边界框
            x_min, x_max = mask_points[:, 0].min(), mask_points[:, 0].max()
            y_min, y_max = mask_points[:, 1].min(), mask_points[:, 1].max()
            
            # 计算面积和长宽比
            width = x_max - x_min
            height = y_max - y_min
            area = width * height
            
            # 计算矩形度（掩码面积与边界框面积的比值）
            mask_area = cv2.contourArea(mask_points.astype(np.float32))
            rectangularity = mask_area / area if area > 0 else 0
            
            # 评分：面积大、矩形度高的优先
            score = area * rectangularity
            
            # 过滤明显不是纸板的目标（长宽比过大或过小）
            aspect_ratio = width / height if height > 0 else 0
            if 2.0 < aspect_ratio < 10.0 and score > best_score:  # 纸板通常是横向的
                best_score = score
                best_idx = idx
        
        # 获取最佳目标的掩码
        mask_points = results[0].masks.xy[best_idx]
        print(f"选择了第 {best_idx + 1} 个检测目标（共 {len(results[0].masks.xy)} 个）")
        
        # 提取上下边缘
        upper_points, lower_points = self.extract_edges(mask_points)
        
        if len(upper_points) < 4 or len(lower_points) < 4:
            print("边缘点不足，无法拟合")
            return None
        
        # 确保上边缘在下边缘上方
        if np.mean(upper_points[:, 1]) > np.mean(lower_points[:, 1]):
            upper_points, lower_points = lower_points, upper_points
            print("交换了上下边缘")
        
        # 拟合曲线（使用多项式）
        upper_curve, upper_params = self.fit_curve(upper_points, method='polynomial', degree=3)
        lower_curve, lower_params = self.fit_curve(lower_points, method='polynomial', degree=3)
        
        # 计算弯曲度
        x_range = [min(upper_points[:, 0].min(), lower_points[:, 0].min()),
                   max(upper_points[:, 0].max(), lower_points[:, 0].max())]
        
        # 获取图像尺寸
        img = Image.open(image_path)
        image_width = img.width
        
        warp_ratio, warp_type = self.calculate_warp(upper_curve, lower_curve, x_range, image_width)
        
        # 判断是否合格
        is_acceptable = warp_ratio < self.warp_threshold
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 原始图像与YOLO检测结果
        ax1 = axes[0, 0]
        ax1.imshow(results[0].plot())
        ax1.set_title('YOLO检测结果', fontsize=12)
        ax1.axis('off')
        
        # 2. 边缘提取与拟合
        ax2 = axes[0, 1]
        ax2.imshow(results[0].orig_img)
        ax2.plot(upper_points[:, 0], upper_points[:, 1], 'r.', markersize=2, label='上边缘点')
        ax2.plot(lower_points[:, 0], lower_points[:, 1], 'b.', markersize=2, label='下边缘点')
        
        # 绘制拟合曲线
        x_fit = np.linspace(x_range[0], x_range[1], 200)
        ax2.plot(x_fit, upper_curve(x_fit), 'r-', linewidth=2, label='上边缘拟合')
        ax2.plot(x_fit, lower_curve(x_fit), 'b-', linewidth=2, label='下边缘拟合')
        ax2.set_title(f'边缘拟合 (弯曲度: {warp_ratio:.4f})', fontsize=12)
        ax2.legend()
        ax2.axis('off')
        
        # 3. 弯曲度分析
        ax3 = axes[1, 0]
        x_sample = np.linspace(x_range[0], x_range[1], 100)
        y_upper = upper_curve(x_sample)
        y_lower = lower_curve(x_sample)
        y_center = (y_upper + y_lower) / 2
        
        # 归一化到0-1范围
        x_norm = (x_sample - x_range[0]) / (x_range[1] - x_range[0])
        
        ax3.plot(x_norm, y_upper, 'r-', label='上边缘')
        ax3.plot(x_norm, y_lower, 'b-', label='下边缘')
        ax3.plot(x_norm, y_center, 'g--', linewidth=2, label='中心线')
        ax3.set_xlabel('归一化位置')
        ax3.set_ylabel('Y坐标（像素）')
        ax3.set_title(f'曲线分析 - 类型: {warp_type}', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.invert_yaxis()
        
        # 4. 梯度分析
        ax4 = axes[1, 1]
        # 计算梯度
        gradient_upper = np.gradient(y_upper)
        gradient_lower = np.gradient(y_lower)
        gradient_center = np.gradient(y_center)
        
        ax4.plot(x_norm[1:], gradient_upper[1:], 'r-', alpha=0.7, label='上边缘梯度')
        ax4.plot(x_norm[1:], gradient_lower[1:], 'b-', alpha=0.7, label='下边缘梯度')
        ax4.plot(x_norm[1:], gradient_center[1:], 'g-', linewidth=2, label='中心线梯度')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.set_xlabel('归一化位置')
        ax4.set_ylabel('梯度')
        ax4.set_title(f'梯度分析 - {"合格" if is_acceptable else "不合格"}', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 添加总标题
        status_text = "✓ 合格" if is_acceptable else "✗ 不合格"
        status_color = 'green' if is_acceptable else 'red'
        fig.suptitle(f'纸板弯曲检测报告 - {status_text}', fontsize=16, color=status_color, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存结果
        if save_results:
            # 保存图像
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            fig_path = os.path.join(output_dir, f'{base_name}_{timestamp}_analysis.png')
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"分析图像已保存: {fig_path}")
            
            # 保存文本报告
            report_path = os.path.join(output_dir, f'{base_name}_{timestamp}_report.txt')
            self.save_report(report_path, image_path, warp_ratio, warp_type, is_acceptable, 
                           upper_params, lower_params, x_range)
            print(f"检测报告已保存: {report_path}")
        
        plt.show()
        
        # 返回结果
        return {
            'image_path': image_path,
            'warp_ratio': warp_ratio,
            'warp_percentage': warp_ratio * 100,
            'warp_type': warp_type,
            'is_acceptable': is_acceptable,
            'threshold': self.warp_threshold,
            'upper_curve_params': upper_params,
            'lower_curve_params': lower_params,
            'edge_points': {
                'upper': upper_points,
                'lower': lower_points
            }
        }
    
    def save_report(self, report_path, image_path, warp_ratio, warp_type, 
                    is_acceptable, upper_params, lower_params, x_range):
        """
        保存检测报告
        """
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("纸板弯曲度检测报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"图像文件: {image_path}\n")
            f.write(f"检测模型: YOLO11-seg\n\n")
            
            f.write("-" * 60 + "\n")
            f.write("检测结果\n")
            f.write("-" * 60 + "\n")
            f.write(f"弯曲度: {warp_ratio:.6f} ({warp_ratio*100:.4f}%)\n")
            f.write(f"弯曲类型: {warp_type}\n")
            f.write(f"阈值: {self.warp_threshold:.4f} ({self.warp_threshold*100:.2f}%)\n")
            f.write(f"检测结果: {'合格 ✓' if is_acceptable else '不合格 ✗'}\n\n")
            
            f.write("-" * 60 + "\n")
            f.write("拟合参数\n")
            f.write("-" * 60 + "\n")
            f.write("上边缘多项式系数 (从高次到低次):\n")
            if upper_params is not None:
                for i, coef in enumerate(upper_params):
                    f.write(f"  x^{len(upper_params)-1-i}: {coef:.6e}\n")
            
            f.write("\n下边缘多项式系数 (从高次到低次):\n")
            if lower_params is not None:
                for i, coef in enumerate(lower_params):
                    f.write(f"  x^{len(lower_params)-1-i}: {coef:.6e}\n")
            
            f.write(f"\nX坐标范围: [{x_range[0]:.1f}, {x_range[1]:.1f}] 像素\n")
            
            f.write("\n" + "-" * 60 + "\n")
            f.write("建议\n")
            f.write("-" * 60 + "\n")
            
            if not is_acceptable:
                f.write("⚠️ 该纸板捆超出弯曲度允许范围，建议:\n")
                f.write("1. 重新调整堆叠方式\n")
                f.write("2. 检查存储环境湿度\n")
                f.write("3. 减少单捆纸板数量\n")
                f.write("4. 使用支撑物防止变形\n")
            else:
                f.write("✓ 该纸板捆符合质量标准\n")
            
            f.write("\n" + "=" * 60 + "\n")

# 使用示例
def main():
    # 初始化检测器
    detector = CardboardWarpDetector(
        model_path='yolo11l-seg.pt',
        warp_threshold=0.02  # 2%的弯曲度阈值
    )
    
    # 检测单张图像
    image_path = 'IMG_5497.JPG'  # 替换为您的图像路径
    
    # 执行检测
    results = detector.detect_warp(
        image_path=image_path,
        save_results=True,
        output_dir='warp_detection_results'
    )
    
    # 打印结果摘要
    if results:
        print("\n" + "=" * 60)
        print("检测摘要")
        print("=" * 60)
        print(f"弯曲度: {results['warp_percentage']:.2f}%")
        print(f"弯曲类型: {results['warp_type']}")
        print(f"检测结果: {'合格' if results['is_acceptable'] else '不合格'}")
        print("=" * 60)
    else:
        print("检测失败，请检查：")
        print("1. YOLO模型是否正确加载")
        print("2. 图像中是否包含纸板")
        print("3. 纸板是否清晰可见")
    
    # 批量处理示例（如果有多张图片）
    # image_list = ['IMG_5497.JPG', 'IMG_5498.JPG', 'IMG_5499.JPG']
    # for img_path in image_list:
    #     results = detector.detect_warp(img_path)
    #     if results:
    #         print(f"{img_path}: {results['warp_percentage']:.2f}% - {'合格' if results['is_acceptable'] else '不合格'}")

if __name__ == "__main__":
    main()