import cv2
import onnx
import numpy as np
import onnxruntime as ort

# output_path = "/circ330/forgithub/VisualFusion_libtorch/Onnx/model/onnxModel/SemLA_onnx_320x240_opencv_compatible.onnx"
output_path = "/circ330/forgithub/VisualFusion_libtorch/Onnx/model/onnxModel/zETOfp16op12_fp16_cuda.onnx"
onnx_model = onnx.load(output_path)

# 範例圖片路徑
img1_path = "/circ330/videodata/Version3/2024-07-10_15-38-54_EO.jpg"
img2_path = "/circ330/videodata/Version3/2024-07-10_15-38-54_IR.jpg"

# 讀取並預處理圖片 (假設模型輸入為320x240, 3通道, BGR)
img1 = cv2.imread(img1_path)
img1 = cv2.resize(img1, (320, 240))
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1 = img1.astype(np.float32) / 255.0

img2 = cv2.imread(img2_path)
img2 = cv2.resize(img2, (320, 240))
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2 = img2.astype(np.float32) / 255.0

img1 = img1[None, None]
img2 = img2[None, None]

print(img1.shape, img2.shape)

# 使用onnxruntime推理
session = ort.InferenceSession(output_path)
# print('================================================')
outputs = session.run(None, {"vi_img": img1, "ir_img": img2})

# exit()
mkpts0, mkpts1,leng1,leng2 = outputs[0], outputs[1],outputs[2],outputs[3]

img1 = cv2.imread(img1_path)
img1 = cv2.resize(img1, (320, 240))

img2 = cv2.imread(img2_path)
img2 = cv2.resize(img2, (320, 240))

"""
mkpts0: numpy.ndarray[N, 2]
mkpts1: numpy.ndarray[N, 2]
"""

mkpts0 = mkpts0.astype(np.float32)[:leng1]
mkpts1 = mkpts1.astype(np.float32)[:leng2]
_, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
mkpts0 = mkpts0[mask.ravel() == 1]
mkpts1 = mkpts1[mask.ravel() == 1]

img = cv2.hconcat([img1, img2])
for (pt0, pt1) in zip(mkpts0, mkpts1):
    x0, y0 = pt0
    x1, y1 = pt1
    
    cv2.circle(img, (int(x0), int(y0)), 2, (0, 255, 0), -1)
    cv2.circle(img, (int(x1) + 320, int(y1)), 2, (0, 0, 255), -1)
    cv2.line(img, (int(x0), int(y0)), (int(x1) + 320, int(y1)), (255, 0, 0), 1)

cv2.imwrite("/circ330/forgithub/VisualFusion_libtorch/convert_to_libtorch/testonnox.jpg", img)
print('img writed!')
    
    
    
    
    