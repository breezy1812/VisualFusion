import torch
import os

from model_jit.SemLA import SemLA

# 使用CPU來避免CUDA相關問題，並提高兼容性
device = torch.device("cpu")
fpMode = torch.float32

print("正在載入模型...")
matcher = SemLA(device=device, fp=fpMode)
matcher.load_state_dict(torch.load(f"./reg.ckpt", map_location=device), strict=False)

matcher = matcher.eval().to(device, dtype=fpMode)

# 使用與配置文件相符的尺寸
width = 320
height = 240

print(f"建立輸入張量，尺寸: {height}x{width}")
torch_input_1 = torch.randn(1, 1, height, width).to(device)
torch_input_2 = torch.randn(1, 1, height, width).to(device)

# if fpMode == torch.float16:
#     torch_input_1 = torch_input_1.half()
#     torch_input_2 = torch_input_2.half()

# 確保輸出目錄存在
output_dir = "/circ330/forgithub/VisualFusion_libtorch/Onnx/model/onnxModel"
os.makedirs(output_dir, exist_ok=True)

output_path = f"{output_dir}/SemLA_onnx_{width}x{height}_opencv_compatible.onnx"

print(f"開始轉換ONNX模型...")
print(f"輸出路徑: {output_path}")

torch.onnx.export(
    matcher,
    (torch_input_1, torch_input_2),
    output_path,
    verbose=False,  # 減少輸出
    opset_version=12,  # 使用支援einsum的版本
    input_names=["vi_img", "ir_img"],
    output_names=["mkpt0", "mkpt1"],  # 簡化輸出
    # 移除dynamic_axes以提高兼容性
    # do_constant_folding=True,  # 啟用常數折疊優化
    dynamic_axes={
        "mkpt0": [0],
        "mkpt1": [0],
    },
)

print("ONNX模型轉換完成！")
print(f"模型已儲存到: {output_path}")

# 驗證模型
try:
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX模型驗證通過")
except Exception as e:
    print(f"⚠️ ONNX模型驗證警告: {e}")

print("🎯 建議更新config.json中的model_path為:")
print(f"    \"{output_path}\"")
