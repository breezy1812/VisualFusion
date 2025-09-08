import os
import torch

from model_jit.SemLA import SemLA

# mode = torch.float16
# device = torch.device("cuda")

# matcher = SemLA(device, mode)
# matcher.load_state_dict(torch.load(f"./reg.ckpt"), strict=False)

# matcher = matcher.eval()

print("=== SemLA ONNX FP16 轉換腳本 (直接導出) ===")

# 使用CUDA來獲得最佳性能
device = torch.device("cuda")
# device = torch.device("cpu")
print(f"使用設備: {device}")

# 直接以 FP16 載入並轉換模型
fpMode = torch.float16
print("正在載入並轉換模型為 FP16...")
matcher = SemLA(device=device, fp=fpMode)
matcher.load_state_dict(torch.load(f"./reg.ckpt", map_location=device), strict=False)
matcher = matcher.eval().to(device, dtype=fpMode)


matcher = torch.jit.script(matcher)

torch.jit.save(matcher, f"/circ330/forgithub/VisualFusion_libtorch/IR_Convert_v21_libtorch/model/SemLA_jit_{device}_fp16.zip")
print('done')