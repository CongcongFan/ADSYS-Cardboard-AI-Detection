from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2

#Instance model
inst_model = YOLO('yolo11l-seg.pt')

# Perform object detection on an image using the model
img = './test_img/More CB Warping.JPG'
instance_results = inst_model.predict(img)

coord = instance_results[0].masks.xy

img_rgb = cv2.cvtColor(instance_results[0].plot(), cv2.COLOR_BGR2RGB)


# 取 YOLO mask 坐标 (示例一条 mask)
coord = instance_results[0].masks.xy[0]  # shape: (N,2)
# 提取所有点
xs, ys = coord[:,0], coord[:,1]

# 找到 mask 的最左 & 最右点，作为 A,B
xmin_idx, xmax_idx = np.argmin(xs), np.argmax(xs)
A = np.array([xs[xmin_idx], ys[xmax_idx]])  # 左边界点
B = np.array([xs[xmax_idx], ys[xmax_idx]])  # 右边界点

# 找到 mask 中最低点 (pyplot 坐标系，数值最大)
lowest_idx = np.argmax(ys)
M = np.array([xs[19], ys[19]])

# 计算 M 到 AB 的垂直距离
AB = B - A
dist = np.abs(np.cross(AB, M - A)) / np.linalg.norm(AB)
L = np.linalg.norm(B - A)

# 半径 & 曲率
h = dist
R = (L**2) / (8*h) + h/2 if h > 0 else np.inf
kappa = 1 / R if R != np.inf else 0

# 像素转毫米
px_per_mm = 2.4
dist_mm = dist / px_per_mm
L_mm = L / px_per_mm
R_mm = R / px_per_mm
warp_pct = (dist_mm / L_mm) * 100
status = "REJECT" if warp_pct >= 2.0 else "PASS"

# 可视化
plt.ylim(3000, 0)
plt.plot(xs, ys, 'r-')
plt.scatter(*A, c='blue', label="A")
plt.scatter(*B, c='blue', label="B")
plt.scatter(*M, c='green', label="Lowest")
plt.legend()
plt.show()

print(f"Deflection h: {dist_mm:.2f} mm")
print(f"Chord length L: {L_mm:.2f} mm")
print(f"Radius R: {R_mm:.2f} mm")
print(f"Warp %: {warp_pct:.2f}%")
print(f"Status: {status}")
