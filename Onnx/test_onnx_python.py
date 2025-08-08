#!/usr/bin/env python3
import onnxruntime as ort
import numpy as np
import cv2

# 測試 ONNX 模型
model_path = "/circ330/forgithub/VisualFusion_libtorch/Onnx/model/onnxModel/SemLA_onnx_320x240_opencv_compatible.onnx"

print("Loading ONNX model...")
session = ort.InferenceSession(model_path)

# 檢查輸入輸出資訊
print("Model inputs:")
for input in session.get_inputs():
    print(f"  {input.name}: {input.shape} ({input.type})")

print("Model outputs:")
for output in session.get_outputs():
    print(f"  {output.name}: {output.shape} ({output.type})")

# 讀取測試圖片
eo_path = "/circ330/circ_video/Version3/0000_eo.png"
ir_path = "/circ330/circ_video/Version3/0000_ir.png"

print(f"Loading test images: {eo_path}, {ir_path}")
eo_img = cv2.imread(eo_path, cv2.IMREAD_GRAYSCALE)
ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

if eo_img is None or ir_img is None:
    print("ERROR: Could not load test images")
    exit(1)

print(f"Original image sizes: EO {eo_img.shape}, IR {ir_img.shape}")

# Resize to model input size
eo_resized = cv2.resize(eo_img, (320, 240))
ir_resized = cv2.resize(ir_img, (320, 240))

print(f"Resized image sizes: EO {eo_resized.shape}, IR {ir_resized.shape}")

# Normalize to [0, 1]
eo_normalized = eo_resized.astype(np.float32) / 255.0
ir_normalized = ir_resized.astype(np.float32) / 255.0

print(f"Normalized ranges: EO [{eo_normalized.min():.3f}, {eo_normalized.max():.3f}], IR [{ir_normalized.min():.3f}, {ir_normalized.max():.3f}]")

# Reshape to [1, 1, H, W]
eo_tensor = eo_normalized.reshape(1, 1, 240, 320)
ir_tensor = ir_normalized.reshape(1, 1, 240, 320)

print(f"Input tensor shapes: EO {eo_tensor.shape}, IR {ir_tensor.shape}")
print(f"First 8 EO pixels: {eo_tensor.flatten()[:8]}")

# Run inference
print("Running inference...")
inputs = {
    "vi_img": eo_tensor,
    "ir_img": ir_tensor
}

outputs = session.run(None, inputs)

print(f"Number of outputs: {len(outputs)}")
for i, output in enumerate(outputs):
    print(f"Output {i} shape: {output.shape}, type: {output.dtype}")
    print(f"Output {i} first 8 values: {output.flatten()[:8]}")
    print(f"Output {i} range: [{output.min():.6f}, {output.max():.6f}]")

print("Python ONNX test completed!")
