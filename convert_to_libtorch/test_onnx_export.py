import cv2
import onnx
import numpy as np
import onnxruntime as ort

print("=== ONNX FP16 模型測試腳本 ===")

# 使用新導出的 FP16 ONNX 模型
output_path = "/circ330/forgithub/VisualFusion_libtorch/Onnx/model/onnxModel/zfp16_op12.onnx"
print(f"載入模型: {output_path}")
onnx_model = onnx.load(output_path)

# 範例圖片路徑
img1_path = "/circ330/videodata/Version3/2024-07-10_15-38-54_EO.jpg"
img2_path = "/circ330/videodata/Version3/2024-07-10_15-38-54_IR.jpg"

# 讀取並預處理圖片
width, height = 320, 240
print(f"預處理圖片，尺寸: {width}x{height}")

img1 = cv2.imread(img1_path)
img1 = cv2.resize(img1, (width, height))
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# 將輸入資料型別改為 float16
img1 = img1.astype(np.float16) / 255.0

img2 = cv2.imread(img2_path)
img2 = cv2.resize(img2, (width, height))
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# 將輸入資料型別改為 float16
img2 = img2.astype(np.float16) / 255.0

img1 = img1[None, None]
img2 = img2[None, None]

print(f"輸入張量維度: {img1.shape}, 資料型別: {img1.dtype}")

# 使用 onnxruntime 推理，並明確指定 CUDA Provider
print("建立 ONNXRuntime 推理 session (CUDAExecutionProvider)...")
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(output_path, providers=providers)

print("執行推理...")
outputs = session.run(None, {"vi_img": img1, "ir_img": img2})

mkpts0, mkpts1, leng1, leng2 = outputs[0], outputs[1], int(outputs[2]), int(outputs[3])
print(f"推理完成，獲取特徵點數量: leng1={leng1}, leng2={leng2}")

# 後處理與視覺化
img1_orig = cv2.imread(img1_path)
img1_orig = cv2.resize(img1_orig, (width, height))

img2_orig = cv2.imread(img2_path)
img2_orig = cv2.resize(img2_orig, (width, height))

# 將特徵點轉回 float32 以便 OpenCV 函式使用
mkpts0 = mkpts0.astype(np.float32)[:leng1]
mkpts1 = mkpts1.astype(np.float32)[:leng2]

print("執行 RANSAC 過濾...")
if len(mkpts0) > 4 and len(mkpts1) > 4:
    _, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
    if mask is not None:
        mask = mask.ravel()
        mkpts0 = mkpts0[mask == 1]
        mkpts1 = mkpts1[mask == 1]
        print(f"RANSAC 後剩餘 {len(mkpts0)} 個匹配點")
    else:
        print("警告: findHomography 未返回遮罩")
else:
    print("特徵點不足，跳過 RANSAC")


img_out = cv2.hconcat([img1_orig, img2_orig])
print("繪製匹配結果...")
for (pt0, pt1) in zip(mkpts0, mkpts1):
    x0, y0 = pt0
    x1, y1 = pt1
    
    cv2.circle(img_out, (int(x0), int(y0)), 2, (0, 255, 0), -1)
    cv2.circle(img_out, (int(x1) + width, int(y1)), 2, (0, 0, 255), -1)
    cv2.line(img_out, (int(x0), int(y0)), (int(x1) + width, int(y1)), (255, 0, 0), 1)

output_image_path = "/circ330/forgithub/VisualFusion_libtorch/convert_to_libtorch/test_onnx_fp16_result.jpg"
cv2.imwrite(output_image_path, img_out)
print(f"✅ 測試完成，結果已儲存至: {output_image_path}")




