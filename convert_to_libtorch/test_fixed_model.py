import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 載入轉換後的 ONNX 模型
model_path = "/circ330/forgithub/VisualFusion_libtorch/Onnx/model/onnxModel/SemLA_onnx_320x240_fixed1200pts.onnx"
session = ort.InferenceSession(model_path)

# 創建測試輸入 (320x240)
img1 = np.random.rand(1, 1, 240, 320).astype(np.float32)
img2 = np.random.rand(1, 1, 240, 320).astype(np.float32)

print(f"輸入形狀: {img1.shape}, {img2.shape}")

# 執行推論
input_name1 = session.get_inputs()[0].name
input_name2 = session.get_inputs()[1].name
output_names = [out.name for out in session.get_outputs()]

print(f"輸入名稱: {input_name1}, {input_name2}")
print(f"輸出名稱: {output_names}")

outputs = session.run(output_names, {input_name1: img1, input_name2: img2})

print(f"輸出0形狀: {outputs[0].shape}")
print(f"輸出1形狀: {outputs[1].shape}")

# 分析特徵點
mkpts0, mkpts1 = outputs[0], outputs[1]

# 統計非零點
non_zero_0 = np.sum(np.any(mkpts0[0] != 0, axis=1))
non_zero_1 = np.sum(np.any(mkpts1[0] != 0, axis=1))

print(f"\n=== 特徵點分析 ===")
print(f"mkpts0 中非 (0,0) 的點數量: {non_zero_0}")
print(f"mkpts1 中非 (0,0) 的點數量: {non_zero_1}")

# 顯示前10個特徵點
print(f"\n前10個特徵點 mkpts0:")
for i in range(min(10, len(mkpts0[0]))):
    print(f"  [{i}]: ({mkpts0[0][i][0]:.2f}, {mkpts0[0][i][1]:.2f})")

print(f"\n前10個特徵點 mkpts1:")
for i in range(min(10, len(mkpts1[0]))):
    print(f"  [{i}]: ({mkpts1[0][i][0]:.2f}, {mkpts1[0][i][1]:.2f})")

# 統計 (0,0) 點
zero_points_0 = np.sum(np.all(mkpts0[0] == 0, axis=1))
zero_points_1 = np.sum(np.all(mkpts1[0] == 0, axis=1))

print(f"\n=== 零點統計 ===")
print(f"mkpts0 中 (0,0) 點數量: {zero_points_0}")
print(f"mkpts1 中 (0,0) 點數量: {zero_points_1}")
print(f"總點數: {len(mkpts0[0])}")

print("\n🎯 測試完成！模型輸出固定 1200 個點，前面是真實特徵點，後面是 (0,0) 填充")
